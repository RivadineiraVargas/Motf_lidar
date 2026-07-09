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
SCALE = 10.0     # normalización de metros para la regresión


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
    """Objetos en frame t + futuros -> dict con pos actuales (ego_t), gt, máscaras."""
    inv = load_pose_inv(scene, t)
    to_ego = lambda p: (inv @ np.append(p, 1.0))[:2]
    cur, gt, wpm = [], [], []
    for f in sorted(glob.glob(f'{ROOT}/objs_bbox/{scene}/{t}/*.txt')):
        tid = os.path.basename(f)[:-4]
        c0 = to_ego(center_of(f))
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
        cur.append(c0); gt.append(np.array(wps)); wpm.append(np.array(mask))
    n = min(len(cur), K_SLOTS)
    return dict(n=n,
                cur=torch.tensor(np.array(cur[:n]), dtype=torch.float32),
                gt=torch.tensor(np.array(gt[:n]), dtype=torch.float32),
                wpm=torch.tensor(np.array(wpm[:n]), dtype=torch.float32))


class MiniWayformerDecoder(nn.Module):
    def __init__(self, enc_dim=384, d=192, heads=4, layers=2):
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(2, d), nn.ReLU(), nn.Linear(d, d))
        self.empty = nn.Parameter(torch.zeros(1, 1, d))
        self.mem_proj = nn.Linear(enc_dim, d)
        layer = nn.TransformerDecoderLayer(d_model=d, nhead=heads,
                                           dim_feedforward=4 * d, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=layers)
        self.head_traj = nn.Linear(d, N_WP * 2)
        self.head_valid = nn.Linear(d, 1)

    def forward(self, mem, cur, n):
        """mem (1,L,enc_dim); cur (n,2) ego/SCALE; n objetos reales."""
        q = self.q_proj(cur.unsqueeze(0))                        # (1,n,d)
        pad = self.empty.expand(1, K_SLOTS - n, -1)
        q = torch.cat([q, pad], dim=1)                           # (1,K,d)
        h = self.dec(q, self.mem_proj(mem))                      # (1,K,d)
        traj = self.head_traj(h).view(1, K_SLOTS, N_WP, 2)
        valid = self.head_valid(h).squeeze(-1)                   # (1,K)
        return traj, valid


@torch.no_grad()
def encode_sweeps(encoder, scene, ts, dev):
    """OJO fork: MAEViT ignora mask=False y siempre enmascara. Con
    mask_ratio=0 conserva TODOS los tokens (permutados, irrelevante para
    cross-attn; el pos embed va antes del shuffle). latent = (1, L+1cls, 384)."""
    old_ratio = encoder.mask_ratio
    encoder.mask_ratio = 0.0
    lat = {}
    for t in ts:
        latent, _, _ = encoder(sweep_tensor(scene, t).to(dev))
        assert latent.shape[-1] == 384 and latent.shape[1] > 6000, \
            f'latent inesperado {tuple(latent.shape)}'
        lat[t] = latent.float()
    encoder.mask_ratio = old_ratio
    return lat


def metrics(traj, valid, s, dev):
    n = s['n']
    pred = traj[0, :n] * SCALE
    gt, wpm = s['gt'].to(dev), s['wpm'].to(dev)
    d = ((pred - gt) ** 2).sum(-1).sqrt()                        # (n,16)
    ade = (d * wpm).sum() / wpm.sum()
    last = wpm.cumsum(1).argmax(1)                               # último wp disponible
    fde = d[torch.arange(n), last].mean()
    lab = torch.zeros(K_SLOTS, device=dev); lab[:n] = 1
    acc = ((valid[0] > 0).float() == lab).float().mean()
    return ade.item(), fde.item(), acc.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', default=['2a81f5233075e987'])
    ap.add_argument('--unseen', default='82f90331a1dfe968')
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default='work_dirs/decoder_mini')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda'
    torch.manual_seed(0)

    init_default_scope('mmpretrain')
    cfg = Config.fromfile(CFG)
    mae = MODELS.build({**cfg.model, 'data_preprocessor': cfg.data_preprocessor})
    sd = torch.load(CKPT, map_location='cpu').get('state_dict')
    mae.load_state_dict(sd, strict=False)
    encoder = mae.backbone.to(dev)
    encoder.eval()          # OJO: este fork retorna None en .eval(), no encadenar
    for p in encoder.parameters():
        p.requires_grad = False

    ts = list(range(11))                                         # t=0..10 (hay LiDAR)
    train_set = []
    for sc in args.scenes:
        lat = encode_sweeps(encoder, sc, ts, dev)
        for t in ts:
            s = build_sample(sc, t)
            if s['n'] > 0:
                train_set.append((lat[t], s))
    lat_u = encode_sweeps(encoder, args.unseen, [10], dev)
    s_u = build_sample(args.unseen, 10)
    print(f'train: {len(train_set)} muestras, objetos medios '
          f'{np.mean([s["n"] for _, s in train_set]):.1f}; unseen n={s_u["n"]}')

    model = MiniWayformerDecoder().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    huber = nn.SmoothL1Loss(reduction='none')
    bce = nn.BCEWithLogitsLoss()

    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for mem, s in train_set:
            n = s['n']
            cur = (s['cur'] / SCALE).to(dev)
            traj, valid = model(mem, cur, n)
            l_traj = (huber(traj[0, :n], (s['gt'] / SCALE).to(dev)).sum(-1)
                      * s['wpm'].to(dev)).sum() / s['wpm'].sum()
            lab = torch.zeros(K_SLOTS, device=dev); lab[:n] = 1
            loss = l_traj + bce(valid[0], lab)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        if ep % 50 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                mem, s = train_set[-1]
                tr = metrics(*model(mem, (s['cur'] / SCALE).to(dev), s['n']), s, dev)
                un = metrics(*model(lat_u[10], (s_u['cur'] / SCALE).to(dev),
                                    s_u['n']), s_u, dev)
            print(f'ep {ep:4d} loss {tot/len(train_set):.4f} | '
                  f'train ADE {tr[0]:.2f} FDE {tr[1]:.2f} acc {tr[2]:.2f} | '
                  f'UNSEEN ADE {un[0]:.2f} FDE {un[1]:.2f} acc {un[2]:.2f}')

    torch.save(model.state_dict(), f'{args.out}/decoder_mini.pth')
    # viz BEV: GT verde, pred rojo, posición actual azul (train t=10 y unseen)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model.eval()
    for name, mem, s in [('train_t10', train_set[-1][0], train_set[-1][1]),
                         ('unseen_t10', lat_u[10], s_u)]:
        with torch.no_grad():
            traj, valid = model(mem, (s['cur'] / SCALE).to(dev), s['n'])
        pred = (traj[0, :s['n']] * SCALE).cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(s['n']):
            c = s['cur'][i].numpy(); g = s['gt'][i].numpy(); m = s['wpm'][i].numpy() > 0
            ax.plot(c[0], c[1], 'b.', ms=6)
            ax.plot(np.r_[c[0], c[0] + g[m, 0]], np.r_[c[1], c[1] + g[m, 1]],
                    'g-', lw=1.5)
            ax.plot(np.r_[c[0], c[0] + pred[i][m, 0]], np.r_[c[1], c[1] + pred[i][m, 1]],
                    'r--', lw=1.2)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_title(f'{name}: GT verde / pred rojo / actual azul (ego, m)')
        fig.savefig(f'{args.out}/bev_{name}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
    print(f'[OK] modelo y BEV guardados en {args.out}/')


if __name__ == '__main__':
    main()
