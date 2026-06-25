"""
viz_mae_reconstruction.py — Visualización de la reconstrucción del encoder MAE
(plano de Claudine, Sec. 5). Por sweep muestra:
  original | enmascarado (75%) | reconstruido (compuesto) | red NO entrenada

Fuerza el enmascaramiento (backbone.train()) y compone: parches visibles = original,
parches ocultos = predicción. Compara contra una red con pesos aleatorios.

Uso: python viz_mae_reconstruction.py [ckpt] [n_sweeps]
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.runner import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmpretrain.datasets.range_view import RangeSweepDataset, unpatchify
from mmpretrain.models.selfsup.mae_4d import MAE4D  # noqa
import mmpretrain.models.backbones.mae_vit_4d  # noqa
import mmpretrain.models.heads.mae_head_4d     # noqa
from mmpretrain.registry import MODELS

CFG = 'configs/sapiens_mae/lidar/mae_sweep_overfit.py'
CKPT = sys.argv[1] if len(sys.argv) > 1 else 'work_dirs/mae_sweep_overfit10/epoch_2000.pth'
N_SHOW = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MAX_RANGE = 75.0
P = 16


def up(patches):
    gh, gw = 64 // P, 512 // P
    return patches.reshape(gh, gw, P, P).transpose(0, 2, 1, 3).reshape(64, 512)


@torch.no_grad()
def recon(model, x):
    """Devuelve (mask, pred) forzando el enmascaramiento."""
    model.backbone.train()
    lat, mask, idr = model.backbone(x)
    pred = model.neck(lat, idr)
    return mask[0].cpu().numpy().astype(bool), pred[0].cpu().numpy()


def main():
    cfg = Config.fromfile(CFG)
    trained = MODELS.build(cfg.model)
    load_checkpoint(trained, CKPT, map_location='cpu'); trained.cuda()
    untrained = MODELS.build(cfg.model).cuda()           # pesos aleatorios

    ds = RangeSweepDataset(data_root='/home/lcad/lidar_sweep_viewer/waymo_clean',
                           max_sweeps=cfg.train_dataloader.dataset.max_sweeps)
    n = min(N_SHOW, len(ds))
    fig, axes = plt.subplots(n, 4, figsize=(20, 2.4 * n))
    if n == 1:
        axes = axes[None, :]

    et, eu = [], []
    for i in range(n):
        x = ds[i]['inputs'].unsqueeze(0).cuda()
        xn = x[0].cpu().numpy()
        torch.manual_seed(i)
        m_t, pred_t = recon(trained, x)
        torch.manual_seed(i)
        m_u, pred_u = recon(untrained, x)

        orig = up(xn) * MAX_RANGE
        masked = xn.copy(); masked[m_t] = np.nan; masked = up(masked) * MAX_RANGE
        comp_t = xn.copy(); comp_t[m_t] = pred_t[m_t]; comp_t = up(comp_t) * MAX_RANGE
        comp_u = xn.copy(); comp_u[m_u] = pred_u[m_u]; comp_u = up(comp_u) * MAX_RANGE
        et.append(((pred_t[m_t]-xn[m_t])**2).mean())
        eu.append(((pred_u[m_u]-xn[m_u])**2).mean())

        for ax, img, t in [(axes[i,0],orig,'original'),(axes[i,1],masked,'enmascarado 75%'),
                           (axes[i,2],comp_t,'reconstruido (entrenado)'),(axes[i,3],comp_u,'red NO entrenada')]:
            ax.imshow(img, cmap='turbo', aspect='auto', origin='lower', vmin=0, vmax=75)
            if i == 0: ax.set_title(t, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f'MAE — encoder pequeno, overfit {len(ds)} sweeps  |  MSE masked: '
                 f'entrenado={np.mean(et):.4f}  vs  no-entrenado={np.mean(eu):.4f}', y=1.0)
    plt.tight_layout()
    plt.savefig('mae_reconstruction.png', dpi=90, bbox_inches='tight')
    print(f'Imagem salva: mae_reconstruction.png')
    print(f'MSE masked  ENTRENADO: {np.mean(et):.4f}   NO-ENTRENADO: {np.mean(eu):.4f}'
          f'   (entrenado debe ser MUCHO menor)')


if __name__ == '__main__':
    main()
