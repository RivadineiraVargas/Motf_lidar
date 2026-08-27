"""viz_un_auto.py — trayectoria de UN objeto: histórico, real y predicha.

Pedido de Claudine: ver la línea de predicción contra la real para un solo auto,
en vez de la vista con decenas de objetos donde no se distingue nada.

Usa los modelos del exp. 16 (entrenados SIN recorte, escala fija), que son los
únicos entrenados sobre la tarea real. Compara las dos variantes que importan:
  gate0 — arquitectura completa, escena APAGADA (la que gana: -10%)
  gated — la misma, con la escena encendida
"""
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.registry import MODELS, DATASETS
init_default_scope('mmpretrain')
import mmpretrain.datasets.trajectory_dataset, mmpretrain.models.backbones.mae_vit_4d
import mmpretrain.models.trajectory_pred.trajectory_model_attn

CFG='configs/sapiens_mae/lidar/noclip_dec_fold0.py'
cfg=Config.fromfile(CFG); dev='cuda'
d=dict(cfg.train_dataloader.dataset)
d.update(scenes=['7e2f727866c69ea0','82f90331a1dfe968'], augment=False, eval_windows=1)
ds=DATASETS.build(d)

def carga(gate_init, freeze, ckpt):
    c=Config.fromfile(CFG); c.model['gate_init']=gate_init; c.model['freeze_gate']=freeze
    m=MODELS.build(c.model); sd=torch.load(ckpt,map_location='cpu')
    m.load_state_dict(sd.get('state_dict',sd),strict=False); m=m.to(dev); m.eval(); return m
M={'gate0': carga(0.0,True ,'work_dirs/noclip/gate0_f0s0/epoch_100.pth'),
   'gated': carga(0.5,False,'work_dirs/noclip/gated_f0s0/epoch_100.pth')}

# tres objetos representativos: lento, medio y rapido (percentiles del desplazamiento)
info=[]
for i in range(len(ds)):
    s_=ds[i]; m_,sd_=s_['norm_mean'].numpy(), s_['norm_std'].numpy()
    g=(s_['obj_future_flat'].reshape(-1,3).numpy()*sd_+m_)[:,:2]
    h=(s_['obj_history_flat'].reshape(-1,3).numpy()*sd_+m_)[:,:2]
    info.append((float(np.linalg.norm(g[-1]-h[-1])), i))
info.sort()
sel=[info[int(len(info)*q)] for q in (0.25,0.55,0.90)]
titulos=['lento (p25)','típico (p55)','rápido (p90)']

fig,axes=plt.subplots(1,3,figsize=(16.5,5.6))
for ax,(desp,i),tit in zip(axes,sel,titulos):
    s_=ds[i]; m_,sd_=s_['norm_mean'].numpy(), s_['norm_std'].numpy()
    hist=(s_['obj_history_flat'].reshape(-1,3).numpy()*sd_+m_)[:,:2]
    gt  =(s_['obj_future_flat'].reshape(-1,3).numpy()*sd_+m_)[:,:2]
    ax.plot(hist[:,0],hist[:,1],'o-',color='#6b7a82',lw=2.2,ms=5,label='histórico (0,5 s)',zorder=3)
    ax.plot(gt[:,0],gt[:,1],'-',color='#2a7570',lw=3.2,label='REAL (3 s)',zorder=2)
    ax.plot(gt[-1,0],gt[-1,1],'*',color='#2a7570',ms=17,zorder=4)
    for k,(col,lab,ls) in {'gate0':('#1f6fb2','sin escena','-'),'gated':('#ab5233','con escena','--')}.items():
        with torch.no_grad():
            pr=M[k](s_['inputs'].unsqueeze(0).to(dev), s_['obj_history_flat'].unsqueeze(0).to(dev), mode='predict').cpu()
        pp=(pr.view(cfg.pred_len,3)*torch.as_tensor(sd_)+torch.as_tensor(m_)).numpy()[:,:2]
        e=np.linalg.norm(pp-gt,axis=1).mean()
        ax.plot(pp[:,0],pp[:,1],ls,color=col,lw=2.4,label=f'{lab} · {e:.1f} m',zorder=2)
        ax.plot(pp[-1,0],pp[-1,1],'o',color=col,ms=8,zorder=4)
    ax.plot(hist[-1,0],hist[-1,1],'o',color='#131a1f',ms=10,zorder=5)
    ax.set_title(f'{tit} — se desplaza {desp:.1f} m', fontsize=11.5, weight='bold')
    ax.set_xlabel('x (m, relativo al ego)'); ax.grid(alpha=.25,ls=':')
    ax.set_aspect('equal',adjustable='datalim'); ax.legend(fontsize=9, loc='best')
axes[0].set_ylabel('y (m)')
fig.suptitle('Trayectoria de un objeto: real contra predicha  ·  ADE por objeto en la leyenda',
             fontsize=13.5, weight='bold', y=1.0)
plt.tight_layout(); plt.savefig('viz_tres_autos.png',dpi=140,bbox_inches='tight')
print('  '+' | '.join(f'{t}: {d:.1f} m' for t,(d,_) in zip(titulos,sel)))
