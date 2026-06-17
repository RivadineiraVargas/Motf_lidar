"""
diag_bbox_lidar.py — Diagnóstico: ¿los bounding boxes coinciden con el LiDAR?

Renderiza BEV (LiDAR + bboxes) de una escena en varios frames, de forma
INDEPENDIENTE del visor C++. Si aquí los boxes SÍ coinciden con los autos, el
problema está en el render del visor; si NO coinciden, está en los datos.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/lcad/lidar_sweep_viewer/waymo_clean'
SCENE = sys.argv[1] if len(sys.argv) > 1 else '7e2f727866c69ea0'
FRAMES = [0, 5, 10]   # frames con LiDAR
VIEW = 40


def load_pose(s, f):
    with open(os.path.join(ROOT, 'poses', s, f'{f}.txt')) as fh:
        rows = [list(map(float, l.split())) for l in fh if len(l.split()) == 4]
    return np.array(rows) if len(rows) == 4 else np.eye(4)


def load_lidar(s, f):
    return np.fromfile(os.path.join(ROOT, 'bin_files', s, f'{f}.bin'),
                       dtype=np.float32).reshape(-1, 4)[:, :3]


def global_to_sensor(verts, pose):
    inv = np.linalg.inv(pose)
    hom = np.hstack([verts, np.ones((len(verts), 1))])
    return (inv @ hom.T).T[:, :3]


def main():
    fig, axes = plt.subplots(1, len(FRAMES), figsize=(6*len(FRAMES), 6))
    for ax, frame in zip(axes, FRAMES):
        pts = load_lidar(SCENE, frame)
        m = (np.abs(pts[:, 0]) < VIEW) & (np.abs(pts[:, 1]) < VIEW)
        pts = pts[m]
        ax.scatter(pts[:, 1], pts[:, 0], s=0.6, c='#888', alpha=.5)

        pose = load_pose(SCENE, frame)
        bbdir = os.path.join(ROOT, 'objs_bbox', SCENE, str(frame))
        n = 0
        if os.path.isdir(bbdir):
            for fn in os.listdir(bbdir):
                if not fn.endswith('.txt'):
                    continue
                with open(os.path.join(bbdir, fn)) as fh:
                    verts = [list(map(float, l.split())) for l in fh if len(l.split()) == 3]
                if len(verts) != 8:
                    continue
                vs = global_to_sensor(np.array(verts), pose)
                base = vs[:4, :2]                      # face inferior (x,y)
                if np.abs(base[:, 0]).max() > VIEW or np.abs(base[:, 1]).max() > VIEW:
                    continue
                poly = plt.Polygon(np.c_[base[:, 1], base[:, 0]], fill=False,
                                   edgecolor='#0ff', lw=1.5)
                ax.add_patch(poly)
                c = base.mean(0)
                ax.text(c[1], c[0], fn[:-4], color='yellow', fontsize=6)
                n += 1
        ax.plot(0, 0, '^', color='red', ms=12)
        ax.set_xlim(-VIEW, VIEW); ax.set_ylim(-VIEW, VIEW); ax.set_aspect('equal')
        ax.set_facecolor('#111'); ax.set_title(f'frame {frame} — {n} bboxes')
        ax.set_xlabel('Y izq (m)'); ax.set_ylabel('X frente (m)')
    plt.suptitle(f'Diagnóstico bbox vs LiDAR — cena {SCENE}', color='k')
    plt.tight_layout()
    out = f'diag_bbox_{SCENE}.png'
    plt.savefig(out, dpi=110, facecolor='white')
    print(f'Guardado: {out}')


if __name__ == '__main__':
    main()
