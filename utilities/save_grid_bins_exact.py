"""
save_grid_bins_exact.py — Extrae bins de GRILLA COMPLETA con xyz EXACTOS
(pipeline oficial del WOMD con pixel_pose = compensación de rolling shutter),
cumpliendo el contrato del viewer C++ como waymo_10:
  64x2650 puntos EN ORDEN de grilla, 4 floats [x, y, z, rango], rango<=0 inválido.

Es el mismo cálculo de womd_lidar_utils.extract_top_lidar_points pero SIN la
máscara final que descarta los píxeles sin retorno (eso fue lo que rompió los
bins de waymo_clean: 153k puntos sueltos que el reshape(64,2650) del C++ no
puede reconstruir).

Uso (en waymo_env):
  conda run -n waymo_env python utilities/save_grid_bins_exact.py \
      --lidar waymo_raw/lidar --out waymo_clean_view/bin_files [--validar]
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils.compression import delta_encoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _load_scenario(path):
    data = next(iter(tf.data.TFRecordDataset(path, compression_type='')))
    return scenario_pb2.Scenario.FromString(data.numpy())


def _calib(frame_lasers, name):
    for c in frame_lasers.laser_calibrations:
        if c.name == name:
            return c
    return None


def grid_frame(laser, frame_pose, calib):
    """(64,2650,4) [x,y,z,rango] de grilla completa, frame del vehículo."""
    pose_arr = delta_encoder.decompress(laser.ri_return1.range_image_pose_delta_compressed)
    rot = transform_utils.get_rotation_matrix(pose_arr[..., 0], pose_arr[..., 1],
                                              pose_arr[..., 2])
    pixel_pose = transform_utils.get_transform(rot, pose_arr[..., 3:])
    pixel_pose = tf.expand_dims(pixel_pose, 0)
    frame_pose_t = tf.expand_dims(tf.convert_to_tensor(frame_pose, tf.float32), 0)

    ri = delta_encoder.decompress(laser.ri_return1.range_image_delta_compressed)
    if not calib.beam_inclinations:
        # mismo helper del oficial (uniforme si el proto no trae la tabla)
        incl = np.linspace(calib.beam_inclination_min, calib.beam_inclination_max,
                           ri.dims[0])
    else:
        incl = np.array(calib.beam_inclinations)
    incl = incl[..., ::-1]
    extrinsic = np.reshape(np.array(calib.extrinsic.transform), [4, 4])

    cart = range_image_utils.extract_point_cloud_from_range_image(
        tf.expand_dims(ri[..., 0], 0),
        tf.expand_dims(extrinsic, 0),
        tf.expand_dims(incl, 0),
        pixel_pose=pixel_pose,
        frame_pose=frame_pose_t)
    cart = tf.squeeze(cart, 0).numpy()                    # (64,2650,3) SIN mascara
    rng = np.array(ri[..., 0])                            # (64,2650) crudo (-1 invalido)
    return np.concatenate([cart, rng[..., None]], axis=-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lidar', default='waymo_raw/lidar')
    ap.add_argument('--out', default='waymo_clean_view/bin_files')
    ap.add_argument('--validar', action='store_true',
                    help='compara vs los bins sueltos de waymo_clean (1 frame)')
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.lidar) if f.endswith('.tfrecord'))
    for fi, f in enumerate(files):
        scene = f[:-9]
        sc = _load_scenario(os.path.join(args.lidar, f))
        os.makedirs(os.path.join(args.out, scene), exist_ok=True)
        t = 0
        for frame_lasers in sc.compressed_frame_laser_data:
            frame_pose = np.reshape(np.array(frame_lasers.pose.transform), (4, 4))
            for laser in frame_lasers.lasers:
                if laser.name != dataset_pb2.LaserName.TOP:
                    continue
                calib = _calib(frame_lasers, dataset_pb2.LaserName.TOP)
                grid = grid_frame(laser, frame_pose, calib)   # (64,2650,4)

                if args.validar:
                    orig = np.fromfile(
                        f'waymo_clean/bin_files/{scene}/{t}.bin',
                        np.float32).reshape(-1, 4)
                    flat = grid.reshape(-1, 4)
                    val = flat[flat[:, 3] > 0]
                    print(f'{scene[:8]} t={t}: grilla {flat.shape[0]} '
                          f'(validos {len(val)}) vs sueltos {len(orig)}')
                    n = min(len(val), len(orig))
                    d = np.linalg.norm(val[:n, :3] - orig[:n, :3], axis=1)
                    print(f'  diff xyz vs extraccion previa (orden): '
                          f'mediana {np.median(d):.4f} m, p99 {np.percentile(d,99):.4f} m')
                    return
                grid.reshape(-1, 4).tofile(os.path.join(args.out, scene, f'{t}.bin'))
            t += 1
        print(f'[{fi+1}/{len(files)}] {scene}: {t} frames')
    print('[OK] bins exactos de grilla completa en', args.out)


if __name__ == '__main__':
    main()
