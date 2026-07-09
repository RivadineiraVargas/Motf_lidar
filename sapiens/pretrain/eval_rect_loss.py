"""
eval_rect_loss.py — Loss L2 enmascarada (la del MAE) por imagen, para comparar
checkpoints/casos cuantitativamente (train/val/no-visto). Misma máscara (seed).

Uso:
  conda run -n sapiens_gpu python eval_rect_loss.py \
      --config configs/... --checkpoint work_dirs/.../epoch_N.pth \
      --inputs img1.png img2.png ... [--seed 0]
"""
import argparse
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.registry import MODELS
from mmcv.transforms import Compose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    init_default_scope('mmpretrain')
    cfg = Config.fromfile(args.config)
    cfg.model['data_preprocessor'] = cfg.data_preprocessor
    model = MODELS.build(cfg.model)
    if args.checkpoint != 'random':
        sd = torch.load(args.checkpoint, map_location='cpu')
        sd = sd.get('state_dict', sd)
        model.load_state_dict(sd, strict=False)
    else:
        model.init_weights()
    model = model.cuda().eval()

    pipe = Compose(cfg.train_pipeline)
    for path in args.inputs:
        data = pipe(dict(img_path=path))
        img = data['inputs'].unsqueeze(0).float().cuda()
        mean = torch.tensor(cfg.data_preprocessor['mean']).view(1, 3, 1, 1).cuda()
        std = torch.tensor(cfg.data_preprocessor['std']).view(1, 3, 1, 1).cuda()
        # PackInputs da BGR; to_rgb=True del preprocessor -> invertir canales
        img = (img[:, [2, 1, 0]] - mean) / std
        torch.manual_seed(args.seed)
        with torch.no_grad():
            latent, mask, ids_restore = model.backbone(img)
            pred = model.neck(latent, ids_restore)
            loss = model.head.loss(pred, img, mask)
        val = loss['loss'].item() if isinstance(loss, dict) else loss.item()
        print(f'{path}: loss={val:.4f}')


if __name__ == '__main__':
    main()
