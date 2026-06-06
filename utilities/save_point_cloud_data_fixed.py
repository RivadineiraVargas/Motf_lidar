"""
save_point_cloud_data_fixed.py
Re-extração CORRIGIDA do WOMD-LiDAR (vs save_point_cloud_data.py do colega).

Dois bugs corrigidos:
  1) ASSOCIAÇÃO DE TRACKS: o original nomeava os bboxes com um contador por frame
     (track_index) que só contava tracks válidos daquele frame -> quando um objeto
     sumia, os índices deslizavam e o "mesmo" id apontava p/ outro carro (saltos de
     dezenas de metros). AQUI usamos `track.id` (ID PERSISTENTE do proto) -> o mesmo
     objeto físico tem sempre o mesmo nome de arquivo em todos os frames.
  2) HORIZONTE: o original tinha `if frame_i < 11` -> só ~1s. AQUI o LiDAR continua
     só nos frames onde existe (~11, limite do WOMD-LiDAR), mas as TRAJETÓRIAS (bbox)
     e poses são salvas p/ TODOS os ~91 frames (9s). Assim dá p/ prever horizonte
     longo (3s/5s/8s) usando 1s de LiDAR como contexto.

Estrutura de saída (compatível com o pipeline atual):
  <root>/bin_files/<scene>/<t>.bin        (LiDAR, só frames com laser ~0..10)
  <root>/poses/<scene>/<t>.txt            (pose 4x4, TODOS os frames)
  <root>/objs_bbox/<scene>/<t>/<track.id>.txt   (8 vértices, TODOS os frames)

Uso:
  conda activate lidar_mae   # tem tensorflow + waymo_open_dataset + open3d
  python save_point_cloud_data_fixed.py \
      --tfrecord <dir_com_os_lidar_{scenario_id}.tfrecord> \
      --scenario <arquivo_scenario.tfrecord> \
      --root <dir_saida> [--max_traj_frames 91]
"""
import os
import struct
import errno
import argparse
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import womd_lidar_utils, box_utils
from waymo_open_dataset.protos import compressed_lidar_pb2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _load_scenario_data(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
    data = next(iter(dataset))
    return scenario_pb2.Scenario.FromString(data.numpy())


def _get_laser_calib(frame_lasers, laser_name):
    for laser_calib in frame_lasers.laser_calibrations:
        if laser_calib.name == laser_name:
            return laser_calib
    return None


def calculate_bounding_box_vertices(track_state):
    box_tensor = tf.constant([[
        track_state.center_x, track_state.center_y, track_state.center_z,
        track_state.length, track_state.width, track_state.height,
        track_state.heading]], dtype=tf.float32)
    corners = box_utils.get_upright_3d_box_corners(box_tensor)
    return corners.numpy()[0]   # (8, 3)


def make_sure_path_exists(caminho):
    try:
        os.makedirs(caminho)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_matrix_txt(mat, filename):
    """Salva uma matriz (ex.: pose 4x4 ou 8 vértices) — uma linha por row."""
    with open(filename, 'w') as f:
        for row in mat:
            if len(row) == 4:
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")
            elif len(row) == 3:
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")


def save_lidar_bin(points_xyzi, bin_path):
    with open(bin_path, 'wb') as fs:   # 'wb' (sobrescreve) — evita acumular se rodar 2x
        for p in points_xyzi:
            fs.write(struct.pack('ffff', p[0], p[1], p[2], p[3]))


def pose_from_state(state):
    """Pose 4x4 (global->? não; é a pose do veículo no mundo) a partir do estado:
    rotação em Z pelo heading + translação pelo centro. Usada p/ os frames de
    trajetória que NÃO têm laser pose (futuro), mantendo o mesmo formato 4x4."""
    h = state.heading
    c, s = np.cos(h), np.sin(h)
    return np.array([
        [c, -s, 0.0, state.center_x],
        [s,  c, 0.0, state.center_y],
        [0.0, 0.0, 1.0, state.center_z],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def process(lidar_dir, proto, bin_path, objs_path, poses_path, max_traj_frames):
    # Fundir o LiDAR (arquivo separado {scenario_id}.tfrecord, com só
    # compressed_frame_laser_data) no scenario proto que tem os tracks.
    # Fluxo oficial: augment(scenario_com_tracks, scenario_com_lidar).
    lidar_file = os.path.join(lidar_dir, proto.scenario_id + '.tfrecord')
    if not os.path.isfile(lidar_file):
        print(f"  [aviso] LiDAR não encontrado p/ {proto.scenario_id}, pulando LiDAR.")
        scenario_aug = proto
    else:
        lidar_scenario = _load_scenario_data(lidar_file)
        scenario_aug = womd_lidar_utils.augment_womd_scenario_with_lidar_points(
            proto, lidar_scenario)

    # ── 1. LiDAR + pose (só frames com laser, ~0..10 = primeiro 1s) ──
    laser_poses = {}   # t -> pose 4x4 (a pose "oficial" do laser p/ esses frames)
    t = 0
    for frame_lasers in scenario_aug.compressed_frame_laser_data:
        frame_pose = np.reshape(np.array(frame_lasers.pose.transform), (4, 4))
        laser_poses[t] = frame_pose
        for laser in frame_lasers.lasers:
            if laser.name == dataset_pb2.LaserName.TOP:
                calib = _get_laser_calib(frame_lasers, dataset_pb2.LaserName.TOP)
                top = womd_lidar_utils.extract_top_lidar_points(laser, frame_pose, calib)
                pts = top[0].numpy()                       # (N,3) sensor frame
                inten = top[1][:, 0].numpy()               # (N,)
                pts_xyzi = np.hstack([pts, inten[:, None]])
                save_lidar_bin(pts_xyzi, os.path.join(bin_path, f"{t}.bin"))
        t += 1
    n_lidar = t
    print(f"  LiDAR: {n_lidar} frames")

    # ── 2. Trajetórias (bbox) + poses p/ TODOS os frames ──
    num_steps = len(proto.timestamps_seconds) if len(proto.timestamps_seconds) else \
        len(proto.tracks[proto.sdc_track_index].states)
    n_traj = min(num_steps, max_traj_frames)
    sdc = proto.tracks[proto.sdc_track_index]

    for t in range(n_traj):
        # pose: usa a do laser se existir (frames LiDAR), senão reconstrói do SDC
        if t in laser_poses:
            pose = laser_poses[t]
        else:
            st = sdc.states[t]
            if not st.valid:
                continue
            pose = pose_from_state(st)
        save_matrix_txt(pose, os.path.join(poses_path, f"{t}.txt"))

        subdir = os.path.join(objs_path, str(t))
        make_sure_path_exists(subdir)
        for track in proto.tracks:
            if t < len(track.states) and track.states[t].valid:
                verts = calculate_bounding_box_vertices(track.states[t])
                # >>> CORREÇÃO PRINCIPAL: nome = track.id (persistente) <<<
                save_matrix_txt(verts, os.path.join(subdir, f"{track.id}.txt"))
    print(f"  Trajetórias: {n_traj} frames (horizonte ~{n_traj/10:.0f}s)")


def main():
    parser = argparse.ArgumentParser(description='Re-extração corrigida WOMD-LiDAR')
    parser.add_argument('-tf', '--tfrecord', required=True,
                        help='Diretório com os LiDAR {scenario_id}.tfrecord')
    parser.add_argument('-s', '--scenario', required=True,
                        help='Arquivo scenario .tfrecord (um shard, contém várias cenas)')
    parser.add_argument('--root', required=True, help='Diretório de saída')
    parser.add_argument('--max_traj_frames', type=int, default=91,
                        help='Máx. de frames de trajetória a salvar (91 = 9s)')
    args = parser.parse_args()

    raw = tf.data.TFRecordDataset([args.scenario])
    n = 0
    for rec in raw:
        proto = scenario_pb2.Scenario()
        proto.ParseFromString(rec.numpy())
        sid = str(proto.scenario_id)
        print(f"[{n}] Scenario {sid}")
        objs = os.path.join(args.root, 'objs_bbox', sid); make_sure_path_exists(objs)
        poses = os.path.join(args.root, 'poses', sid);    make_sure_path_exists(poses)
        binp = os.path.join(args.root, 'bin_files', sid); make_sure_path_exists(binp)
        process(args.tfrecord, proto, binp, objs, poses, args.max_traj_frames)
        n += 1
    print(f"Pronto: {n} cenas extraídas em {args.root}")


if __name__ == '__main__':
    main()
