"""
list_scene_ids.py — Lista los scenario_id contenidos en un shard de scenario.

Sirve para saber qué archivos LiDAR ({scenario_id}.tfrecord) hay que descargar
para fusionarlos con ese shard.

Uso:
    conda activate waymo_env
    python list_scene_ids.py <shard.tfrecord> [--limit N]

Imprime un scenario_id por línea (los primeros N si se pasa --limit).
"""
import sys
import argparse
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shard', help='Archivo scenario .tfrecord (un shard)')
    ap.add_argument('--limit', type=int, default=0,
                    help='Solo los primeros N scene_ids (0 = todos)')
    args = ap.parse_args()

    raw = tf.data.TFRecordDataset([args.shard])
    n = 0
    for rec in raw:
        proto = scenario_pb2.Scenario()
        proto.ParseFromString(rec.numpy())
        print(proto.scenario_id)
        n += 1
        if args.limit and n >= args.limit:
            break


if __name__ == '__main__':
    main()
