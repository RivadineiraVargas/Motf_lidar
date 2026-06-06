import sys; sys.path.insert(0, '.')
import torch
from mmpretrain.registry import MODELS

cfg = dict(
    type='TrajectoryModelWithAttention',
    encoder=dict(type='MAEViT4D', history_len=5, embed_dim=1024, num_tokens=300,
                 arch='sapiens_0.3b', final_norm=True, mask_ratio=0.75),
    history_len=5, pred_len=5, embed_dim=1024, num_heads=8, hidden_dim=512,
)
model = MODELS.build(cfg)
ckpt = torch.load('work_dirs/mae_encoder_pretrained.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)
print("Pesos cargados OK — patch_embed std:", round(float(model.encoder.patch_embed.weight.std()), 4))

x    = torch.randn(1, 300, 5)
hist = torch.randn(1, 15)
fut  = torch.randn(1, 15)
loss_dict = model(x, hist, mode='loss', obj_future_flat=fut)
loss_dict['loss'].backward()

enc_grad  = model.encoder.patch_embed.weight.grad
head_grad = model.decoder[0].weight.grad

print("Gradiente encoder.patch_embed:", "OK" if enc_grad is not None else "NONE (sin gradiente!)")
if enc_grad is not None:
    print("  norm:", round(float(enc_grad.norm()), 6))
print("Gradiente decoder[0]:", "OK" if head_grad is not None else "NONE")
if head_grad is not None:
    print("  norm:", round(float(head_grad.norm()), 6))

if enc_grad is not None and head_grad is not None:
    ratio = float(head_grad.norm()) / (float(enc_grad.norm()) + 1e-10)
    print("Ratio grad_decoder/grad_encoder:", round(ratio, 1), "x")
    if ratio > 1000:
        print("PROBLEMA: gradientes del encoder son ~0 — encoder no está aprendiendo")
    else:
        print("Gradientes OK")

# Número de parámetros por componente
enc_p  = sum(p.numel() for p in model.encoder.parameters())
head_p = sum(p.numel() for p in model.parameters()) - enc_p
print(f"\nParams encoder: {enc_p:,} ({enc_p*100//(enc_p+head_p)}%)")
print(f"Params head:    {head_p:,} ({head_p*100//(enc_p+head_p)}%)")
