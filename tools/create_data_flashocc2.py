#!/usr/bin/env python3
"""Generate FlashOCC NuScenes info pkl files in one pass.

Each sample is processed once and directly produces a full info dict, including:
- lidar/camera paths and transforms
- sweep info
- detection training targets (ann_infos)
- occupancy path metadata (scene_name, occ_path)

Outputs:
- <out_dir>/<extra_tag>_infos_train.pkl
- <out_dir>/<extra_tag>_infos_val.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from tqdm import tqdm


MAP_NAME_FROM_GENERAL_TO_DETECTION = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}

CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

CAMERA_TYPES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


_WORKER_NUSC: NuScenes | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create NuScenes info pkl for FlashOCC")
    parser.add_argument("--root-path", default="data/nuScenes", help="NuScenes root path")
    parser.add_argument("--out-dir", default=None, help="Output directory for generated pkl")
    parser.add_argument("--extra-tag", default="flashocc2-nuscenes", help="Output file prefix")
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        choices=["v1.0-trainval", "v1.0-test", "v1.0-mini"],
        help="NuScenes version",
    )
    parser.add_argument("--max-sweeps", type=int, default=0, help="Max previous lidar sweeps")
    parser.add_argument("--occ-root", default=None, help="Occupancy GT root path (default: <root-path>/gts)")
    parser.add_argument(
        "--path-mode",
        default="relative",
        choices=["relative", "absolute"],
        help="How to store paths in pkl",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) // 4)),
        help="Workers for sample-level parallelism (set 1 to disable)",
    )
    parser.add_argument(
        "--parallel-backend",
        default="thread",
        choices=["thread", "process"],
        help="thread is memory-friendly; process may be faster but uses more memory",
    )
    return parser.parse_args()


def normalize_path(path: Path, cwd: Path, mode: str) -> str:
    if mode == "absolute":
        return str(path.resolve())
    try:
        rel = path.resolve().relative_to(cwd)
        return "./" + rel.as_posix()
    except ValueError:
        return str(path.resolve())


def get_split_scene_names(version: str) -> Tuple[List[str], List[str], bool]:
    if version == "v1.0-trainval":
        return list(splits.train), list(splits.val), False
    if version == "v1.0-test":
        return list(splits.test), [], True
    if version == "v1.0-mini":
        return list(splits.mini_train), list(splits.mini_val), False
    raise ValueError(f"Unsupported version: {version}")


def get_available_scene_tokens(nusc: NuScenes) -> Dict[str, str]:
    scene_name_to_token: Dict[str, str] = {}
    print(f"total scene num: {len(nusc.scene)}")
    for scene in nusc.scene:
        first_sample = nusc.get("sample", scene["first_sample_token"])
        lidar_sd = nusc.get("sample_data", first_sample["data"]["LIDAR_TOP"])
        lidar_path = Path(nusc.get_sample_data_path(lidar_sd["token"]))
        if lidar_path.exists():
            scene_name_to_token[scene["name"]] = scene["token"]
    print(f"exist scene num: {len(scene_name_to_token)}")
    return scene_name_to_token


def _init_worker(version: str, dataroot: str) -> None:
    global _WORKER_NUSC
    _WORKER_NUSC = NuScenes(version=version, dataroot=dataroot, verbose=False)


def _get_nusc_for_worker(shared_nusc: NuScenes | None) -> NuScenes:
    if shared_nusc is not None:
        return shared_nusc
    if _WORKER_NUSC is None:
        raise RuntimeError("NuScenes worker is not initialized")
    return _WORKER_NUSC


def obtain_sensor2top(
    nusc: NuScenes,
    sensor_token: str,
    l2e_t: np.ndarray,
    l2e_r_mat: np.ndarray,
    e2g_t: np.ndarray,
    e2g_r_mat: np.ndarray,
    sensor_type: str,
    cwd: Path,
    path_mode: str,
) -> Dict:
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = normalize_path(Path(nusc.get_sample_data_path(sd_rec["token"])), cwd, path_mode)

    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }

    l2e_r_s_mat = Quaternion(sweep["sensor2ego_rotation"]).rotation_matrix
    e2g_r_s_mat = Quaternion(sweep["ego2global_rotation"]).rotation_matrix

    r_s_to_l = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    t_s_to_l = (
        np.array(sweep["sensor2ego_translation"]) @ e2g_r_s_mat.T + np.array(sweep["ego2global_translation"])
    ) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    t_s_to_l -= np.array(e2g_t) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    t_s_to_l -= np.array(l2e_t) @ np.linalg.inv(l2e_r_mat).T

    sweep["sensor2lidar_rotation"] = r_s_to_l.T
    sweep["sensor2lidar_translation"] = t_s_to_l
    return sweep


def build_detection_targets(cam_front: Dict, ann_infos_raw: List[Dict]) -> Tuple[List[np.ndarray], List[int]]:
    ego2global_rotation = cam_front["ego2global_rotation"]
    ego2global_translation = cam_front["ego2global_translation"]
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse

    gt_boxes: List[np.ndarray] = []
    gt_labels: List[int] = []

    for ann_info in ann_infos_raw:
        det_name = MAP_NAME_FROM_GENERAL_TO_DETECTION.get(ann_info["category_name"], "ignore")
        if det_name not in CLASSES:
            continue
        if ann_info["num_lidar_pts"] + ann_info["num_radar_pts"] <= 0:
            continue

        box = Box(
            ann_info["translation"],
            ann_info["size"],
            Quaternion(ann_info["rotation"]),
            velocity=ann_info["velocity"],
        )
        box.translate(trans)
        box.rotate(rot)

        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_boxes.append(np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo]))
        gt_labels.append(CLASSES.index(det_name))

    return gt_boxes, gt_labels


def build_single_info(
    nusc: NuScenes,
    sample_token: str,
    max_sweeps: int,
    test: bool,
    cwd: Path,
    path_mode: str,
    occ_root: Path,
) -> Tuple[str, Dict]:
    sample = nusc.get("sample", sample_token)

    lidar_token = sample["data"]["LIDAR_TOP"]
    sd_rec = nusc.get("sample_data", lidar_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])

    info = {
        "lidar_path": normalize_path(Path(nusc.get_sample_data_path(lidar_token)), cwd, path_mode),
        "token": sample_token,
        "scene_token": sample["scene_token"],
        "sweeps": [],
        "cams": {},
        "lidar2ego_translation": cs_record["translation"],
        "lidar2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sample["timestamp"],
    }

    l2e_t = np.array(info["lidar2ego_translation"])
    l2e_r_mat = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    e2g_t = np.array(info["ego2global_translation"])
    e2g_r_mat = Quaternion(info["ego2global_rotation"]).rotation_matrix

    for cam in CAMERA_TYPES:
        cam_token = sample["data"][cam]
        _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
        cam_info = obtain_sensor2top(
            nusc,
            cam_token,
            l2e_t,
            l2e_r_mat,
            e2g_t,
            e2g_r_mat,
            cam,
            cwd,
            path_mode,
        )
        cam_info["cam_intrinsic"] = cam_intrinsic
        info["cams"][cam] = cam_info

    sweeps = []
    sweep_sd = sd_rec
    while len(sweeps) < max_sweeps and sweep_sd["prev"] != "":
        sweep = obtain_sensor2top(
            nusc,
            sweep_sd["prev"],
            l2e_t,
            l2e_r_mat,
            e2g_t,
            e2g_r_mat,
            "lidar",
            cwd,
            path_mode,
        )
        sweeps.append(sweep)
        sweep_sd = nusc.get("sample_data", sweep_sd["prev"])
    info["sweeps"] = sweeps

    if not test:
        _, boxes, _ = nusc.get_sample_data(lidar_token)
        annotations = [nusc.get("sample_annotation", ann_token) for ann_token in sample["anns"]]

        if len(annotations) > 0:
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
            velocity = np.array([nusc.box_velocity(ann_token)[:2] for ann_token in sample["anns"]])
            valid_flag = np.array(
                [(anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0 for anno in annotations], dtype=bool
            ).reshape(-1)

            for idx in range(len(boxes)):
                velo = np.array([*velocity[idx], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[idx] = velo[:2]

            names = np.array([MAP_NAME_FROM_GENERAL_TO_DETECTION.get(ann["category_name"], ann["category_name"]) for ann in annotations])
            info["gt_boxes"] = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array([ann["num_lidar_pts"] for ann in annotations])
            info["num_radar_pts"] = np.array([ann["num_radar_pts"] for ann in annotations])
            info["valid_flag"] = valid_flag
        else:
            info["gt_boxes"] = np.zeros((0, 7), dtype=np.float32)
            info["gt_names"] = np.array([], dtype=object)
            info["gt_velocity"] = np.zeros((0, 2), dtype=np.float32)
            info["num_lidar_pts"] = np.array([], dtype=np.int64)
            info["num_radar_pts"] = np.array([], dtype=np.int64)
            info["valid_flag"] = np.array([], dtype=bool)

        ann_infos_raw = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            velocity = nusc.box_velocity(ann["token"])
            if np.any(np.isnan(velocity)):
                velocity = np.zeros(3)
            ann["velocity"] = velocity
            ann_infos_raw.append(ann)

        info["ann_infos"] = build_detection_targets(info["cams"]["CAM_FRONT"], ann_infos_raw)

        scene = nusc.get("scene", sample["scene_token"])
        info["scene_name"] = scene["name"]
        info["occ_path"] = normalize_path(occ_root / scene["name"] / sample_token, cwd, path_mode)

    return sample["scene_token"], info


def _build_single_info_worker(task: Tuple[str, int, bool, str, str, str]) -> Tuple[str, Dict]:
    sample_token, max_sweeps, test, cwd_str, path_mode, occ_root_str = task
    nusc = _get_nusc_for_worker(shared_nusc=None)
    return build_single_info(
        nusc=nusc,
        sample_token=sample_token,
        max_sweeps=max_sweeps,
        test=test,
        cwd=Path(cwd_str),
        path_mode=path_mode,
        occ_root=Path(occ_root_str),
    )


def iter_sample_infos(
    nusc: NuScenes,
    sample_tokens: List[str],
    max_sweeps: int,
    test: bool,
    cwd: Path,
    path_mode: str,
    occ_root: Path,
    num_workers: int,
    parallel_backend: Literal["thread", "process"],
) -> Iterable[Tuple[str, Dict]]:
    num_workers = max(1, int(num_workers))

    if num_workers == 1:
        for token in tqdm(sample_tokens, desc="Building infos"):
            yield build_single_info(nusc, token, max_sweeps, test, cwd, path_mode, occ_root)
        return

    tasks = [(token, max_sweeps, test, str(cwd), path_mode, str(occ_root)) for token in sample_tokens]

    if parallel_backend == "process":
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(nusc.version, str(Path(nusc.dataroot).resolve())),
        ) as pool:
            mapped = pool.map(_build_single_info_worker, tasks, chunksize=32)
            for result in tqdm(mapped, total=len(tasks), desc=f"Building infos (process:{num_workers})"):
                yield result
        return

    global _WORKER_NUSC
    _WORKER_NUSC = nusc
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        mapped = pool.map(_build_single_info_worker, tasks, chunksize=32)
        for result in tqdm(mapped, total=len(tasks), desc=f"Building infos (thread:{num_workers})"):
            yield result


def dump_infos(out_path: Path, infos: List[Dict], version: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump({"infos": infos, "metadata": {"version": version}}, f)


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    root_path = Path(args.root_path).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root_path
    occ_root = Path(args.occ_root).resolve() if args.occ_root else (root_path / "gts")

    if not root_path.exists():
        raise FileNotFoundError(f"NuScenes root does not exist: {root_path}")

    print(f"[create_data] root_path={root_path}")
    print(f"[create_data] out_dir={out_dir}")
    print(f"[create_data] extra_tag={args.extra_tag}")
    print(f"[create_data] version={args.version}")
    print(f"[create_data] max_sweeps={args.max_sweeps}")
    print(f"[create_data] occ_root={occ_root}")
    print(f"[create_data] path_mode={args.path_mode}")
    print(f"[create_data] num_workers={max(1, args.num_workers)}")
    print(f"[create_data] parallel_backend={args.parallel_backend}")

    nusc = NuScenes(version=args.version, dataroot=str(root_path), verbose=True)
    train_scene_names, val_scene_names, test = get_split_scene_names(args.version)
    scene_name_to_token = get_available_scene_tokens(nusc)

    train_scene_tokens = {scene_name_to_token[name] for name in train_scene_names if name in scene_name_to_token}
    val_scene_tokens = {scene_name_to_token[name] for name in val_scene_names if name in scene_name_to_token}
    print(f"train scene: {len(train_scene_tokens)}, val scene: {len(val_scene_tokens)}")

    sample_tokens = [sample["token"] for sample in nusc.sample]
    train_infos: List[Dict] = []
    val_infos: List[Dict] = []

    for scene_token, info in iter_sample_infos(
        nusc=nusc,
        sample_tokens=sample_tokens,
        max_sweeps=args.max_sweeps,
        test=test,
        cwd=cwd,
        path_mode=args.path_mode,
        occ_root=occ_root,
        num_workers=args.num_workers,
        parallel_backend=args.parallel_backend,
    ):
        if scene_token in train_scene_tokens:
            train_infos.append(info)
        elif scene_token in val_scene_tokens:
            val_infos.append(info)

    if test:
        test_path = out_dir / f"{args.extra_tag}_infos_test.pkl"
        dump_infos(test_path, train_infos, args.version)
        print(f"[create_data] test samples: {len(train_infos)}")
        print(f"[create_data] wrote: {test_path}")
        return

    train_path = out_dir / f"{args.extra_tag}_infos_train.pkl"
    val_path = out_dir / f"{args.extra_tag}_infos_val.pkl"
    dump_infos(train_path, train_infos, args.version)
    dump_infos(val_path, val_infos, args.version)

    print(f"[create_data] train samples: {len(train_infos)}")
    print(f"[create_data] val samples: {len(val_infos)}")
    print(f"[create_data] wrote: {train_path}")
    print(f"[create_data] wrote: {val_path}")


if __name__ == "__main__":
    main()
