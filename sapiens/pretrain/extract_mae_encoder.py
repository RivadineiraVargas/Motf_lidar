"""
extract_mae_encoder.py — Extrae el encoder de un checkpoint MAE para los
trajectory models.

El checkpoint MAE (MAE4D) guarda 'backbone.*' (el ViT) + 'neck.*' (el decoder
de reconstrucción). Los trajectory models cargan un encoder con keys 'encoder.*'.
Este script toma las keys 'backbone.*', las renombra a 'encoder.*', descarta el
neck, y guarda el resultado.

Uso:
    conda activate sapiens_gpu
    cd sapiens/pretrain
    python extract_mae_encoder.py \
        work_dirs/mae_clean25/epoch_1000.pth \
        work_dirs/mae_encoder_clean25.pth
"""
import sys
import torch


def main():
    if len(sys.argv) != 3:
        print('Uso: python extract_mae_encoder.py <mae_ckpt.pth> <salida_encoder.pth>')
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)

    new_sd = {}
    n_back, n_skip = 0, 0
    for k, v in sd.items():
        if k.startswith('backbone.'):
            new_sd['encoder.' + k[len('backbone.'):]] = v
            n_back += 1
        else:
            n_skip += 1   # neck, etc.

    torch.save({'state_dict': new_sd}, dst)
    print(f'Encoder extraído: {n_back} keys backbone->encoder, {n_skip} descartadas')
    print(f'Guardado en: {dst}')


if __name__ == '__main__':
    main()
