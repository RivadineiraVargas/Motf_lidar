"""
train_decoder_mini.py — Mini-fase 2 (Claudine pasos 13-15, 17) a escala 10-lambidas.

Decoder estilo WAYFORMER CONDICIONADO (ver docs/ESTUDIO_WAYFORMER.md):
  - Encoder MAE (rect, reducido) CONGELADO -> tokens de escena del sweep actual.
  - K=100 slots de trayectoria; los primeros N condicionados en la posición
    actual (ego) de cada objeto movible; el resto, embedding "vacío" aprendido.
  - 2 bloques TransformerDecoder (self-attn entre slots + cross-attn a escena).
  - Cabezas: 16 waypoints (desplazamientos ego, 8s a 2Hz, formato WOMD) + flag
    de validez por slot (BCE).
  - Loss: Huber sobre waypoints disponibles + BCE de validez.
  - Métricas: ADE/FDE (m) + accuracy de validez, en train y en escena NO vista.

Datos: waymo_clean (labels 0..90 = 9s; LiDAR 0..10). Muestra = (escena, t<=10):
sweep t como entrada, futuro t+5..t+80 (paso 5 frames = 2Hz) desde objs_bbox.

Uso:
  conda run -n sapiens_gpu python train_decoder_mini.py \
      --scenes 2a81f5233075e987 --unseen 82f90331a1dfe968 \
      --epochs 500 --out work_dirs/decoder_mini
"""
import argparse, os, glob
import numpy as np
import torch
import torch.nn as nn
import cv2
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.registry import MODELS

ROOT = '/home/lcad/lidar_sweep_viewer/waymo_clean'
CFG = 'configs/sapiens_mae/lidar/config_rangeview_rect_overfit10.py'
CKPT = 'work_dirs/rv_rect_overfit10/epoch_6000.pth'
MAXR = 75.0
K_SLOTS = 100
N_WP = 16        # 16 waypoints x 0.5s = 8s (formato WOMD)
WP_STEP = 5      # frames entre waypoints (10Hz -> 2Hz)
H_PAST = 10      # frames de historia a 10Hz = 1.0s (máximo con LiDAR en t<=10;
                 # el WOMD-LiDAR solo trae ~1.1s de LiDAR por escena)
SCALE = 10.0     # normalización de metros para la regresión
FEAT_DIM = 2 + 2 * H_PAST   # query: pos actual + desplazamientos pasados


def sweep_tensor(scene, t):
    """npy (64,2650,2) -> tensor normalizado (1,3,1024,2650) como en el MAE."""
    r = np.load(f'{ROOT}/range_files/{scene}/{t}.npy')[..., 0]
    u = np.clip(255 * (1 - r / MAXR), 0, 255).astype(np.uint8)
    u[r <= 0] = 0
    img = cv2.resize(u, (2650, 1024), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    x = np.stack([img] * 3)                     # canales idénticos (gris)
    mean = np.array([123.675, 116.28, 103.53], np.float32).reshape(3, 1, 1)
    std = np.array([58.395, 57.12, 57.375], np.float32).reshape(3, 1, 1)
    return torch.from_numpy((x - mean) / std).unsqueeze(0)


def load_pose_inv(scene, t):
    m = np.loadtxt(f'{ROOT}/poses/{scene}/{t}.txt')
    return np.linalg.inv(m)


def center_of(path):
    return np.loadtxt(path).mean(axis=0)        # media de 8 esquinas (global)


def build_sample(scene, t):
    """Objetos en frame t + historia + futuros -> dict con pos actuales
    (ego_t), features de query (pos + historia 1.0s), gt y máscaras."""
    inv = load_pose_inv(scene, t)
    to_ego = lambda p: (inv @ np.append(p, 1.0))[:2]
    cur, feat, gt, wpm, cv, ids = [], [], [], [], [], []
    for f in sorted(glob.glob(f'{ROOT}/objs_bbox/{scene}/{t}/*.txt')):
        tid = os.path.basename(f)[:-4]
        c0 = to_ego(center_of(f))
        # historia: desplazamientos pos(t-j)-pos(t) a 10Hz (estilo Wayformer:
        # la query conoce velocidad y rumbo). Si falta un frame pasado se
        # repite el último disponible (clamp).
        hist, last = [], c0
        for j in range(1, H_PAST + 1):
            fp = f'{ROOT}/objs_bbox/{scene}/{max(t - j, 0)}/{tid}.txt'
            if os.path.exists(fp):
                last = to_ego(center_of(fp))
            hist.append(last - c0)
        wps, mask = [], []
        for k in range(1, N_WP + 1):
            ff = f'{ROOT}/objs_bbox/{scene}/{t + k * WP_STEP}/{tid}.txt'
            if os.path.exists(ff):
                wps.append(to_ego(center_of(ff)) - c0)   # desplazamiento ego
                mask.append(1.0)
            else:
                wps.append(np.zeros(2)); mask.append(0.0)
        if sum(mask) == 0:
            continue                                     # sin futuro: se omite
        # extrapolacion velocidad constante (piso cinematico): v de los
        # ultimos 0.5s; el modelo predice el RESIDUO sobre esto
        v = -hist[4] / 0.5
        cvp = np.stack([v * (0.5 * (k + 1)) for k in range(N_WP)])
        cur.append(c0); gt.append(np.array(wps)); wpm.append(np.array(mask))
        feat.append(np.concatenate([c0, np.concatenate(hist)]))
        cv.append(cvp)
        ids.append(tid)
    n = min(len(cur), K_SLOTS)
    return dict(n=n, ids=ids[:n],
                cur=torch.tensor(np.array(cur[:n]), dtype=torch.float32),
                feat=torch.tensor(np.array(feat[:n]), dtype=torch.float32),
                cv=torch.tensor(np.array(cv[:n]), dtype=torch.float32),
                gt=torch.tensor(np.array(gt[:n]), dtype=torch.float32),
                wpm=torch.tensor(np.array(wpm[:n]), dtype=torch.float32))


class MiniBaseline(nn.Module):
    """Referencia SIN escena: MLP por slot sobre la posición actual. Misma
    interfaz (mem se ignora) para comparar modelos (Claudine Sec.11)."""

    def __init__(self, d=192):
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(FEAT_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.empty = nn.Parameter(torch.zeros(1, 1, d))
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.head_traj = nn.Linear(d, N_WP * 2)
        nn.init.zeros_(self.head_traj.weight)   # residuo arranca en 0 = piso CV
        nn.init.zeros_(self.head_traj.bias)
        self.head_valid = nn.Linear(d, 1)

    def forward(self, mem, cur, n):
        q = self.q_proj(cur.unsqueeze(0))
        q = torch.cat([q, self.empty.expand(1, K_SLOTS - n, -1)], dim=1)
        h = self.mlp(q)
        return (self.head_traj(h).view(1, K_SLOTS, N_WP, 2),
                self.head_valid(h).squeeze(-1))


class MiniWayformerDecoder(nn.Module):
    def __init__(self, enc_dim=384, d=192, heads=4, layers=2, max_hist=8):
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(FEAT_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.empty = nn.Parameter(torch.zeros(1, 1, d))
        self.mem_proj = nn.Linear(enc_dim, d)
        # embedding temporal por sweep de historia (Sec.1 Claudine: entrada
        # multi-sweep); indice 0 = sweep actual, 1 = anterior, etc.
        self.t_emb = nn.Parameter(torch.zeros(max_hist, d))
        layer = nn.TransformerDecoderLayer(d_model=d, nhead=heads,
                                           dim_feedforward=4 * d, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=layers)
        self.head_traj = nn.Linear(d, N_WP * 2)
        nn.init.zeros_(self.head_traj.weight)   # residuo arranca en 0 = piso CV
        nn.init.zeros_(self.head_traj.bias)
        self.head_valid = nn.Linear(d, 1)

    def forward(self, mem, cur, n):
        """mem: tensor (1,L,enc_dim) o lista [actual, t-1, ...]; cur (n,FEAT_DIM)
        = [pos, hist]/SCALE ego; n objetos reales."""
        if not isinstance(mem, (list, tuple)):
            mem = [mem]
        mems = [self.mem_proj(m) + self.t_emb[i] for i, m in enumerate(mem)]
        mem = torch.cat(mems, dim=1)                             # (1,k*L,d)
        q = self.q_proj(cur.unsqueeze(0))                        # (1,n,d)
        pad = self.empty.expand(1, K_SLOTS - n, -1)
        q = torch.cat([q, pad], dim=1)                           # (1,K,d)
        h = self.dec(q, mem)                                     # (1,K,d)
        traj = self.head_traj(h).view(1, K_SLOTS, N_WP, 2)
        valid = self.head_valid(h).squeeze(-1)                   # (1,K)
        return traj, valid


class MiniWayformerPooled(nn.Module):
    """Wayformer con PUENTE LIVIANO: en vez de que las K=100 queries del
    decoder atiendan directamente a los ~6784 tokens crudos del encoder MAE
    (mucho ruido/capacidad para overfitear con pocas escenas — ver hallazgo
    de la validación cruzada del 29/07), se resume cada sweep con `n_pool`
    latentes aprendidos (estilo Perceiver / latent queries de Wayformer,
    ver docs/ESTUDIO_WAYFORMER.md) ANTES de la cross-attention final."""

    def __init__(self, enc_dim=384, d=192, heads=4, layers=2, max_hist=8,
                n_pool=16):
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(FEAT_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.empty = nn.Parameter(torch.zeros(1, 1, d))
        self.mem_proj = nn.Linear(enc_dim, d)
        self.t_emb = nn.Parameter(torch.zeros(max_hist, d))
        # latentes de pooling: compartidos entre sweeps, resumen ~6784 -> n_pool
        self.pool_latents = nn.Parameter(torch.randn(1, n_pool, d) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        layer = nn.TransformerDecoderLayer(d_model=d, nhead=heads,
                                           dim_feedforward=4 * d, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=layers)
        self.head_traj = nn.Linear(d, N_WP * 2)
        nn.init.zeros_(self.head_traj.weight)   # residuo arranca en 0 = piso CV
        nn.init.zeros_(self.head_traj.bias)
        self.head_valid = nn.Linear(d, 1)

    def forward(self, mem, cur, n):
        if not isinstance(mem, (list, tuple)):
            mem = [mem]
        pooled = []
        for i, m in enumerate(mem):
            mp = self.mem_proj(m) + self.t_emb[i]                # (1, L, d)
            lat = self.pool_latents.expand(mp.shape[0], -1, -1)
            out, _ = self.pool_attn(lat, mp, mp)                 # (1, n_pool, d)
            pooled.append(out)
        mem_pooled = torch.cat(pooled, dim=1)                    # (1, k*n_pool, d)
        q = self.q_proj(cur.unsqueeze(0))
        pad = self.empty.expand(1, K_SLOTS - n, -1)
        q = torch.cat([q, pad], dim=1)
        h = self.dec(q, mem_pooled)
        traj = self.head_traj(h).view(1, K_SLOTS, N_WP, 2)
        valid = self.head_valid(h).squeeze(-1)
        return traj, valid


class GatedDecoderLayer(nn.Module):
    """Equivalente a nn.TransformerDecoderLayer (post-norm), pero con la
    contribución de la CROSS-ATTENTION (la rama de escena) escalada por un
    gate. Hay que escribirla a mano: en la capa de PyTorch la cross-attn
    está fusionada adentro y no se puede escalar por separado.

    El gate NO lo posee la capa — llega por `forward`, compartido por todas,
    para que sea UN escalar por modelo (igual que en Fase 1) y no uno por capa."""

    def __init__(self, d, heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, mem, g):
        a, _ = self.self_attn(x, x, x)
        x = self.n1(x + a)
        c, _ = self.cross_attn(x, mem, mem)
        x = self.n2(x + g * c)                  # <-- GATE sobre la rama de escena
        return self.n3(x + self.ff(x))


class MiniWayformerGated(nn.Module):
    """Wayformer + GATE aprendible sobre la escena — tercer ingrediente de la
    Fase 1 (los otros dos, encoder adaptado al dominio y horizonte 3s, se
    replicaron el 06/08: −20.4%, p=0.0006 en el fold 0).

    Escalar único aprendido, aplicado como tanh(scene_gate), igual que el
    modelo de vóxeles de Fase 1 (mmpretrain/models/trajectory_pred/
    trajectory_model_attn.py). Deja que el modelo aprenda CUÁNTO condicionar
    en la escena en vez de obligarlo a usarla siempre: el diagnóstico viejo
    (escena 9e89) era que sobre-corregía justo donde la velocidad constante
    ya era perfecta.

    gate_init=0.5, NO 0: arrancar en 0 anula el gradiente de toda la rama de
    escena y el gate no abre nunca — candado documentado en Fase 1.

    Bonus de interpretabilidad: tanh(scene_gate) al final del entrenamiento
    es una medida directa de cuánto decidió el modelo apoyarse en la escena."""

    def __init__(self, enc_dim=384, d=192, heads=4, layers=2, max_hist=8,
                 gate_init=0.5):
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(FEAT_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.empty = nn.Parameter(torch.zeros(1, 1, d))
        self.mem_proj = nn.Linear(enc_dim, d)
        self.t_emb = nn.Parameter(torch.zeros(max_hist, d))
        gate_init = float(max(min(gate_init, 0.99), -0.99))
        self.scene_gate = nn.Parameter(torch.atanh(torch.tensor([gate_init])))
        self.blocks = nn.ModuleList([GatedDecoderLayer(d, heads)
                                     for _ in range(layers)])
        self.head_traj = nn.Linear(d, N_WP * 2)
        nn.init.zeros_(self.head_traj.weight)   # residuo arranca en 0 = piso CV
        nn.init.zeros_(self.head_traj.bias)
        self.head_valid = nn.Linear(d, 1)

    def forward(self, mem, cur, n):
        if not isinstance(mem, (list, tuple)):
            mem = [mem]
        mems = [self.mem_proj(m) + self.t_emb[i] for i, m in enumerate(mem)]
        mem = torch.cat(mems, dim=1)                             # (1,k*L,d)
        q = self.q_proj(cur.unsqueeze(0))                        # (1,n,d)
        pad = self.empty.expand(1, K_SLOTS - n, -1)
        h = torch.cat([q, pad], dim=1)                           # (1,K,d)
        g = torch.tanh(self.scene_gate)
        for blk in self.blocks:
            h = blk(h, mem, g)
        return (self.head_traj(h).view(1, K_SLOTS, N_WP, 2),
                self.head_valid(h).squeeze(-1))


@torch.no_grad()
def encode_sweeps(encoder, scene, ts, dev, cache_dir=None):
    """OJO fork: MAEViT ignora mask=False y siempre enmascara. Con
    mask_ratio=0 conserva TODOS los tokens (permutados, irrelevante para
    cross-attn: es un set no-ordenado consumido por cross-attention, el
    orden no afecta el resultado). latent = (1, L+1cls, 384).

    cache_dir: si se da, cachea en disco por escena (todo `ts` junto) para
    no recodificar la misma escena en cada corrida (clave para validación
    cruzada: 25 escenas se codifican UNA vez, se reusan en fold x seed x arch).
    """
    if cache_dir is not None:
        path = os.path.join(cache_dir, f'{scene}.pt')
        if os.path.exists(path):
            cached = torch.load(path, map_location=dev)
            if all(t in cached for t in ts):
                return {t: cached[t] for t in ts}

    old_ratio = encoder.mask_ratio
    encoder.mask_ratio = 0.0
    lat = {}
    for t in ts:
        latent, _, _ = encoder(sweep_tensor(scene, t).to(dev))
        assert latent.shape[-1] == 384 and latent.shape[1] > 6000, \
            f'latent inesperado {tuple(latent.shape)}'
        lat[t] = latent.float()
    encoder.mask_ratio = old_ratio

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({t: v.cpu() for t, v in lat.items()}, path)
    return lat


def metrics(traj, valid, s, dev):
    n = s['n']
    pred = s['cv'].to(dev) + traj[0, :n] * SCALE   # cv + residuo aprendido
    gt, wpm = s['gt'].to(dev), s['wpm'].to(dev)
    d = ((pred - gt) ** 2).sum(-1).sqrt()                        # (n, N_WP)
    ade = (d * wpm).sum() / wpm.sum()                            # ADE @ horizonte
    # "ade5" = ADE al mínimo entre 10 wp (5s) y el horizonte actual, para que
    # el barrido de horizonte con N_WP<10 no rompa el slice
    k5 = min(10, N_WP)
    ade5 = (d[:, :k5] * wpm[:, :k5]).sum() / wpm[:, :k5].sum()
    last = wpm.cumsum(1).argmax(1)                               # último wp disponible
    fde = d[torch.arange(n), last].mean()
    lab = torch.zeros(K_SLOTS, device=dev); lab[:n] = 1
    acc = ((valid[0] > 0).float() == lab).float().mean()
    return ade.item(), ade5.item(), fde.item(), acc.item()


def load_frozen_encoder(enc_ckpt, dev):
    init_default_scope('mmpretrain')
    cfg = Config.fromfile(CFG)
    mae = MODELS.build({**cfg.model, 'data_preprocessor': cfg.data_preprocessor})
    sd = torch.load(enc_ckpt, map_location='cpu').get('state_dict')
    mae.load_state_dict(sd, strict=False)
    encoder = mae.backbone.to(dev)
    encoder.eval()          # OJO: este fork retorna None en .eval(), no encadenar
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def unfreeze_encoder_tail(encoder, n_blocks=1):
    """Descongela los últimos n_blocks bloques transformer del encoder MAE
    (+ la norma final) para fine-tuning parcial junto al decoder. El resto
    queda congelado -> conserva la representación pre-entrenada, solo
    permite que las capas finales se adapten a la tarea de trayectorias.
    Exploración: ¿el problema era el encoder congelado, no el puente?
    (ver hallazgo 29/07: ni atención cruda ni pooling ayudaron con
    features 100% congeladas)."""
    total = len(encoder.layers)
    assert 0 < n_blocks <= total, f'n_blocks debe estar en (0,{total}]'
    for i, block in enumerate(encoder.layers):
        req = i >= total - n_blocks
        for p in block.parameters():
            p.requires_grad = req
    for p in encoder.ln1.parameters():
        p.requires_grad = True
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f'[fine-tune] descongelados los últimos {n_blocks}/{total} bloques '
          f'+ ln1 ({n_trainable:,} params entrenables de '
          f'{sum(p.numel() for p in encoder.parameters()):,})')
    return encoder


def encode_one_live(encoder, scene, t, dev):
    """Como encode_sweeps pero UN sweep, SIN @torch.no_grad() (el caller
    controla el contexto de gradiente) y SIN cache (el encoder cambia
    entre épocas cuando se está fine-tuneando -> cachear sería usar
    features obsoletas de una época anterior, un bug silencioso)."""
    old_ratio = encoder.mask_ratio
    encoder.mask_ratio = 0.0
    latent, _, _ = encoder(sweep_tensor(scene, t).to(dev))
    encoder.mask_ratio = old_ratio
    return latent.float()


def train_decoder(scenes, unseen, epochs=500, lr=1e-3, arch='wayformer', hist=1,
                  enc_ckpt=CKPT, out_dir='work_dirs/decoder_mini', seed=0,
                  encoder=None, cache_dir=None, eval_every=20, save_viz=True,
                  verbose=True, dev='cuda', finetune_encoder_blocks=0, enc_lr=1e-5,
                  n_wp=None):
    """Entrena el decoder mini y devuelve (best_ade8, best_ep, history).
    Única fuente de verdad del loop de entrenamiento — usada tanto por el
    CLI (main) como por cross_validate_decoder.py, para que ambos caminos
    nunca diverjan.

    encoder: si se pasa (ya cargado), se reusa en vez de recargar el
    checkpoint — ahorra I/O cuando se llama muchas veces (validación cruzada).

    finetune_encoder_blocks: si >0, descongela los últimos N bloques del
    encoder MAE y los entrena junto al decoder (lr propio, más bajo:
    enc_lr). Fuerza cache_dir=None (el encoder cambia entre épocas; cachear
    sería servir features obsoletas). Exploración post-29/07: ¿el problema
    era el encoder 100% congelado, no el diseño del puente?

    n_wp: horizonte de predicción en waypoints (2/6/10/16 = 1/3/5/8s).
    Fija el global N_WP ANTES de construir samples y modelos (build_sample,
    los head_traj de los 3 modelos y metrics leen ese global). Contenido:
    se setea explícito al inicio de cada llamada, así corridas secuenciales
    con distinto horizonte no se contaminan.
    """
    global N_WP
    if n_wp is not None:
        N_WP = n_wp
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    if encoder is None:
        encoder = load_frozen_encoder(enc_ckpt, dev)
    finetuning = finetune_encoder_blocks > 0
    if finetuning:
        unfreeze_encoder_tail(encoder, finetune_encoder_blocks)
        if cache_dir is not None and verbose:
            print('[fine-tune] cache_dir ignorado (encoder no está congelado)')
        cache_dir = None

    ts = list(range(11))                                         # t=0..10 (hay LiDAR)
    hist_of = lambda lat, t: [lat[max(t - i, 0)] for i in range(hist)]  # clamp t=0
    train_set = []
    for sc in scenes:
        if not finetuning:
            lat = encode_sweeps(encoder, sc, ts, dev, cache_dir=cache_dir)
        for t in ts:
            s = build_sample(sc, t)
            if s['n'] > 0:
                # fine-tuning: guardamos (escena,t) y recodificamos cada
                # época (con grad); frozen: guardamos el latent ya calculado
                mem = (sc, t) if finetuning else hist_of(lat, t)
                train_set.append((mem, s))
    unseen_evals = []
    for usc in unseen:
        if not finetuning:
            lat_u = encode_sweeps(encoder, usc, ts, dev, cache_dir=cache_dir)
        mem_u_entry = (usc, 10) if finetuning else hist_of(lat_u, 10)
        unseen_evals.append((mem_u_entry, build_sample(usc, 10)))
    mem_u, s_u = unseen_evals[0]
    if verbose:
        print(f'train: {len(train_set)} muestras, objetos medios '
              f'{np.mean([s["n"] for _, s in train_set]):.1f}; '
              f'unseen escenas={len(unseen_evals)}')

    live_mem = lambda sc, t, grad: (
        [encode_one_live(encoder, sc, max(t - i, 0), dev) for i in range(hist)]
        if grad else
        [encode_one_live(encoder, sc, max(t - i, 0), dev).detach() for i in range(hist)])

    model = {'wayformer': MiniWayformerDecoder,
             'wayformer_pooled': MiniWayformerPooled,
             'wayformer_gated': MiniWayformerGated,
             'baseline': MiniBaseline}[arch]().to(dev)
    gate_of = (lambda: float(torch.tanh(model.scene_gate).item())) \
        if hasattr(model, 'scene_gate') else (lambda: None)
    best_ade, best_ep = float('inf'), 0
    suffix = '' if arch == 'wayformer' else f'_{arch}'
    best_path = f'{out_dir}/decoder_mini{suffix}.pth'
    best_metrics = None
    param_groups = [{'params': model.parameters(), 'lr': lr}]
    if finetuning:
        enc_params = [p for p in encoder.parameters() if p.requires_grad]
        param_groups.append({'params': enc_params, 'lr': enc_lr})
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-2)
    huber = nn.SmoothL1Loss(reduction='none')
    bce = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        model.train()
        if finetuning:
            encoder.train()          # OJO fork: retorna None, no encadenar
        tot = 0.0
        for mem, s in train_set:
            n = s['n']
            if finetuning:
                sc, t = mem
                mem = live_mem(sc, t, grad=True)
            cur = (s['feat'] / SCALE).to(dev)
            traj, valid = model(mem, cur, n)
            l_traj = (huber(traj[0, :n], ((s['gt'] - s['cv']) / SCALE).to(dev)).sum(-1)
                      * s['wpm'].to(dev)).sum() / s['wpm'].sum()
            lab = torch.zeros(K_SLOTS, device=dev); lab[:n] = 1
            loss = l_traj + bce(valid[0], lab)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if finetuning:
                nn.utils.clip_grad_norm_(enc_params, 1.0)
            opt.step()
            tot += loss.item()
        if ep % eval_every == 0 or ep == 1:
            model.eval()
            if finetuning:
                encoder.eval()       # OJO fork: retorna None, no encadenar
            with torch.no_grad():
                mem, s = train_set[-1]
                if finetuning:
                    sc, t = mem
                    mem = live_mem(sc, t, grad=False)
                tr = metrics(*model(mem, (s['feat'] / SCALE).to(dev), s['n']), s, dev)
                uns = []
                for mu, su in unseen_evals:
                    if finetuning:
                        usc, ut = mu
                        mu = live_mem(usc, ut, grad=False)
                    uns.append(metrics(*model(mu, (su['feat'] / SCALE).to(dev),
                                              su['n']), su, dev))
                un = [sum(x) / len(x) for x in zip(*uns)]
            marca = ''
            if un[0] < best_ade:
                best_ade, best_ep, best_metrics = un[0], ep, dict(
                    ade8=un[0], ade5=un[1], fde=un[2], acc=un[3],
                    train_ade8=tr[0], gate=gate_of())
                torch.save(model.state_dict(), best_path)
                if finetuning:
                    torch.save({k: v for k, v in encoder.state_dict().items()},
                               f'{out_dir}/encoder_tail{suffix}.pth')
                marca = '  [mejor]'
            if verbose:
                g = gate_of()
                print(f'ep {ep:4d} loss {tot/len(train_set):.4f} | '
                      f'train ADE8 {tr[0]:.2f} ADE5 {tr[1]:.2f} FDE {tr[2]:.2f} '
                      f'acc {tr[3]:.2f} | UNSEEN ADE8 {un[0]:.2f} ADE5 {un[1]:.2f} '
                      f'FDE {un[2]:.2f} acc {un[3]:.2f}'
                      + (f' | gate {g:+.3f}' if g is not None else '') + marca)

    # gate al FINAL del entrenamiento, no solo en el mejor checkpoint: si el
    # early-stop se queda en una época temprana, el gate del checkpoint casi no
    # se movio de su init y no dice nada sobre a donde converge la decision.
    if best_metrics is not None:
        best_metrics['gate_final'] = gate_of()

    if verbose:
        gf = gate_of()
        print(f'[early-stop] mejor checkpoint: ep {best_ep} '
              f'(UNSEEN ADE8 {best_ade:.2f}) -> {best_path}'
              + (f' | gate final {gf:+.3f}' if gf is not None else ''))

    if save_viz:
        model.load_state_dict(torch.load(best_path, map_location=dev))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        model.eval()
        if finetuning:
            encoder.eval()           # OJO fork: retorna None, no encadenar
        for name, mem, s in [('train_t10', train_set[-1][0], train_set[-1][1]),
                             ("unseen_t10", mem_u, s_u)]:
            with torch.no_grad():
                if finetuning:
                    sc, t = mem
                    mem = live_mem(sc, t, grad=False)
                traj, valid = model(mem, (s['feat'] / SCALE).to(dev), s['n'])
            pred = (s['cv'].to(dev) + traj[0, :s['n']] * SCALE).cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 10))
            for i in range(s['n']):
                c = s['cur'][i].numpy(); g = s['gt'][i].numpy()
                m = s['wpm'][i].numpy() > 0
                ax.plot(c[0], c[1], 'b.', ms=6)
                ax.plot(np.r_[c[0], c[0] + g[m, 0]], np.r_[c[1], c[1] + g[m, 1]],
                        'g-', lw=1.5)
                ax.plot(np.r_[c[0], c[0] + pred[i][m, 0]], np.r_[c[1], c[1] + pred[i][m, 1]],
                        'r--', lw=1.2)
            ax.set_aspect('equal'); ax.grid(alpha=0.3)
            ax.set_title(f'{name}: GT verde / pred rojo / actual azul (ego, m)')
            fig.savefig(f'{out_dir}/bev_{name}.png', dpi=120, bbox_inches='tight')
            plt.close(fig)
        if verbose:
            print(f'[OK] modelo y BEV guardados en {out_dir}/')

    return best_metrics, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', default=['2a81f5233075e987'])
    ap.add_argument('--unseen', nargs='+', default=['82f90331a1dfe968'])
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--arch', choices=['wayformer', 'wayformer_pooled', 'baseline'],
                    default='wayformer')
    ap.add_argument('--hist', type=int, default=1,
                    help='k sweeps de historia como entrada (Sec.1 Claudine)')
    ap.add_argument('--enc', default=CKPT,
                    help='checkpoint del encoder MAE congelado')
    ap.add_argument('--out', default='work_dirs/decoder_mini')
    ap.add_argument('--finetune-blocks', type=int, default=0,
                    help='descongelar los últimos N bloques del encoder '
                         '(0 = frozen, comportamiento validado)')
    ap.add_argument('--enc-lr', type=float, default=1e-5)
    ap.add_argument('--eval-every', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    train_decoder(args.scenes, args.unseen, epochs=args.epochs, lr=args.lr,
                 arch=args.arch, hist=args.hist, enc_ckpt=args.enc,
                 out_dir=args.out, seed=args.seed, eval_every=args.eval_every,
                 finetune_encoder_blocks=args.finetune_blocks,
                 enc_lr=args.enc_lr)


if __name__ == '__main__':
    main()
