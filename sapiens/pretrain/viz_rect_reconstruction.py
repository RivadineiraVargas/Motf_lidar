"""
viz_rect_reconstruction.py — Visualización de reconstrucción MAE para el pipeline
RECTANGULAR (estilo colega, panorámico). Genera 4 paneles apilados verticalmente:
original | enmascarado | reconstruido | reconstruido+visible.

No depende de test_dataloader (construye el modelo a mano). Uso:
  conda run -n sapiens_gpu python viz_rect_reconstruction.py \
      --config configs/sapiens_mae/lidar/config_rangeview_rect_overfit10.py \
      --checkpoint work_dirs/rv_rect_overfit10/epoch_2000.pth \
      --input /ruta/a/sweep_000.png --output recon_out/rect_train000.png
"""
import argparse
import numpy as np
import torch
import cv2
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.registry import MODELS
from mmcv.transforms import Compose


def to_img(t):
    """(3,H,W) normalizado -> uint8 HxW (gris) en [0,255]."""
    a = t.detach().cpu().numpy().transpose(1, 2, 0)  # H,W,3
    a = np.clip(a, 0, 255).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', default=None,
                    help='ruta al .pth; "random" = red SIN entrenar (Claudine Sec.5)')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', default='recon_rect.png')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    init_default_scope('mmpretrain')
    cfg = Config.fromfile(args.config)
    cfg.model['data_preprocessor'] = cfg.data_preprocessor

    torch.manual_seed(args.seed)
    model = MODELS.build(cfg.model)
    model.init_weights()
    if args.checkpoint and args.checkpoint != 'random':
        sd = torch.load(args.checkpoint, map_location='cpu')
        sd = sd.get('state_dict', sd)
        model.load_state_dict(sd, strict=False)
    else:
        print('[AVISO] red SIN entrenar (pesos de inicializacion)')
    model = model.cuda().eval()

    # cargar imagen con el MISMO pipeline del entrenamiento (Resize bicubic)
    pipeline = Compose([t for t in cfg.train_pipeline])
    data = pipeline({'img_path': args.input})
    inp = data['inputs'].unsqueeze(0).float().cuda()  # (1,3,H,W)
    mean = model.data_preprocessor.mean
    std = model.data_preprocessor.std
    x = (inp - mean) / std

    head = model.head
    with torch.no_grad():
        latent, mask, ids_restore = model.backbone(x)
        pred = model.neck(latent, ids_restore)              # (1, L, ph*pw*3)
        target = head.patchify(x)                            # (1, L, ph*pw*3)

    # imágenes en espacio de patches
    recon_patches = pred
    # recon+visible: usar original en los visibles, pred en los enmascarados
    m = mask.unsqueeze(-1)                                    # (1,L,1) 1=enmascarado
    recon_vis_patches = target * (1 - m) + pred * m
    masked_patches = target * (1 - m)                        # enmascarados a 0

    # volver a imagen y des-normalizar
    def unpatch_denorm(p):
        img = head.unpatchify(p)                             # (1,3,H,W) normalizado
        return img * std + mean

    orig_img = unpatch_denorm(target)[0]
    masked_img = unpatch_denorm(masked_patches)[0]
    recon_img = unpatch_denorm(recon_patches)[0]
    reconvis_img = unpatch_denorm(recon_vis_patches)[0]

    # panel de DIFERENCIA (Claudine Sec.5): |original - (recon+visible)|, amplificada
    # x3 para que el error sea visible; blanco = error grande, negro = sin error.
    diff = (orig_img - reconvis_img).abs() * 3.0
    diff_gray = to_img(diff.clamp(0, 255))        # error claro sobre negro

    panels = [to_img(orig_img), to_img(masked_img), to_img(recon_img),
              to_img(reconvis_img), diff_gray]
    sep = np.full((6, panels[0].shape[1]), 255, np.uint8)    # separador blanco
    stacked = panels[0]
    for p in panels[1:]:
        stacked = np.vstack([stacked, sep, p])
    cv2.imwrite(args.output, stacked)
    # también cada panel suelto
    for name, p in zip(['original', 'masked', 'recon', 'recon_plus_visible', 'diff'], panels):
        cv2.imwrite(args.output.replace('.png', f'_{name}.png'), p)
    print(f'[OK] guardado {args.output} (5 paneles: original/enmascarado/recon/recon+visible/diff)')


if __name__ == '__main__':
    main()
