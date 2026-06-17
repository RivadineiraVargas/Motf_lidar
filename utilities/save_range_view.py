"""
save_range_view.py — Extrae la RANGE-VIEW NATIVA (64 beams x cols x [range,intensidad])
del WOMD-LiDAR, para el track range-view del MOTF.

A diferencia de save_point_cloud_data_fixed.py (que convierte a puntos cartesianos),
aquí guardamos la imagen de rango tal cual la mide el sensor — densa y con la
estructura exacta de 64 beams. Salida: range_files/<scene>/<t>.npy  (64, W, 2).

Uso (en waymo_env):
  python utilities/save_range_view.py --tfrecord waymo_raw/lidar/ \
      --scenario waymo_raw/scenario/training.tfrecord-00000-of-01000 --root waymo_clean/
"""
import os, argparse
import numpy as np
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import womd_lidar_utils as w


def make_sure(d):
    os.makedirs(d, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-tf', '--tfrecord', required=True, help='dir con {scene}.tfrecord (lidar)')
    ap.add_argument('-s', '--scenario', required=True, help='shard scenario .tfrecord')
    ap.add_argument('--root', required=True, help='dir de salida (crea range_files/)')
    args = ap.parse_args()

    lidar_ids = {f.replace('.tfrecord', '') for f in os.listdir(args.tfrecord)
                 if f.endswith('.tfrecord')}
    out_root = os.path.join(args.root, 'range_files')
    n_scene = 0

    for rec in tf.data.TFRecordDataset([args.scenario]):
        proto = scenario_pb2.Scenario()
        proto.ParseFromString(rec.numpy())
        sid = proto.scenario_id
        if sid not in lidar_ids:
            continue

        # fusionar lidar
        lidar = scenario_pb2.Scenario()
        for r in tf.data.TFRecordDataset(
                [os.path.join(args.tfrecord, sid + '.tfrecord')], compression_type=''):
            lidar.ParseFromString(r.numpy()); break
        aug = w.augment_womd_scenario_with_lidar_points(proto, lidar)

        scene_dir = os.path.join(out_root, sid)
        make_sure(scene_dir)
        t = 0
        for fl in aug.compressed_frame_laser_data:
            for laser in fl.lasers:
                if laser.name == dataset_pb2.LaserName.TOP:
                    ri = np.array(w.delta_encoder.decompress(
                        laser.ri_return1.range_image_delta_compressed))
                    # canales 0=range, 1=intensidad
                    out = ri[..., :2].astype(np.float32)
                    np.save(os.path.join(scene_dir, f'{t}.npy'), out)
            t += 1
        n_scene += 1
        print(f'  {sid}: {t} frames range-view ({out.shape})')

    print(f'Listo: {n_scene} escenas -> {out_root}')


if __name__ == '__main__':
    main()
