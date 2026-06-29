"""
Convert raw ScanNet++ iPhone data to the RE10K-style .torch chunk format
used by the DatasetRE10k loader.

Actual raw ScanNet++ directory layout:

    <INPUT_DIR>/
      splits/
        sem_test.txt           # one scene ID per line (used for "test" stage)
        nvs_sem_train.txt      # used for "train" stage
        ...
      data/
        <scene_id>/
          iphone/
            rgb.mkv            # video of all frames (1920x1440)
            pose_intrinsic_imu.json  # per-frame aligned_pose (C2W) + intrinsic (3x3 K)

Output layout matches DatasetRE10k expectations:

    <OUTPUT_DIR>/
      test/
        000000.torch
        ...
        index.json

Each .torch file is a list of scene dicts:
    {
        "key":        str,                    # e.g. "be2e10f16a_iphone"
        "url":        str,
        "timestamps": Int64Tensor[N],
        "cameras":    Float32Tensor[N, 18],   # [fx/w, fy/h, cx/w, cy/h, w, h, W2C_3x4_flat]
        "images":     list[UInt8Tensor[...]],  # raw JPEG bytes
        "depths":     list,                    # empty list (depth not packed)
    }

Camera row layout (18 floats):
    cols  0-3 : fx/w, fy/h, cx/w, cy/h  (normalized intrinsics)
    cols  4-5 : image width, height       (pixel dimensions, after any resize)
    cols 6-17 : W2C 3x4 matrix, row-major (aligned_pose inverted)
"""

import argparse
import json
import multiprocessing as mp
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from jaxtyping import Float, UInt8
from torch import Tensor
from tqdm import tqdm

TARGET_BYTES_PER_CHUNK = int(1e8)  # ~100 MB per chunk
JPEG_QUALITY = 95

STAGE_TO_SPLIT_FILE = {
    "train": "nvs_sem_train.txt",
    "test": "sem_test.txt",
}


@dataclass
class ConvertConfig:
    input_dir: Path
    output_dir: Path
    width: int
    height: int
    min_dist: float = 0.05  # motion-based: min camera translation (m) between kept frames


def get_scene_ids(input_dir: Path, stage: Literal["train", "test"]) -> list[str]:
    """Return scene IDs for the given split.

    Resolution order:
      1. <input_dir>/<stage>.txt          (user override, one ID per line)
      2. <input_dir>/splits/<*.txt>       (official split files)
      3. All directories under <input_dir>/data/  (fallback)
    """
    override = input_dir / f"{stage}.txt"
    if override.exists():
        return [l.strip() for l in override.read_text().splitlines() if l.strip()]

    official = input_dir / "splits" / STAGE_TO_SPLIT_FILE[stage]
    if official.exists():
        return [l.strip() for l in official.read_text().splitlines() if l.strip()]

    return sorted(d.name for d in (input_dir / "data").iterdir() if d.is_dir())


def _select_by_motion(meta: dict, all_keys: list[str], min_dist: float) -> set[int]:
    """Return indices of frames to keep based on minimum camera translation.

    Greedily keeps a frame only when the camera has moved at least min_dist
    metres from the last kept frame. Always keeps the first frame.
    """
    wanted = {0}
    last_pos = np.array(meta[all_keys[0]]["aligned_pose"])[:3, 3]
    for i, key in enumerate(all_keys[1:], 1):
        pos = np.array(meta[key]["aligned_pose"])[:3, 3]
        if np.linalg.norm(pos - last_pos) >= min_dist:
            wanted.add(i)
            last_pos = pos
    return wanted


def load_frames_and_cameras(
    iphone_dir: Path,
    cfg: ConvertConfig,
    tqdm_position: int = 0,
) -> tuple[list[UInt8[Tensor, "..."]], Float[Tensor, "N 18"]] | None:
    """Decode selected frames from rgb.mkv and build the [N, 18] camera tensor.

    Frame selection is either stride-based (every Nth frame) or motion-based
    (minimum camera translation between kept frames), depending on cfg.
    Frames are resized to (cfg.width, cfg.height) before JPEG encoding.
    Returns None if the scene should be skipped.
    """
    video_path = iphone_dir / "rgb.mkv"
    json_path = iphone_dir / "pose_intrinsic_imu.json"

    if not video_path.exists() or not json_path.exists():
        return None

    with open(json_path) as fh:
        meta = json.load(fh)

    all_keys = sorted(meta.keys())

    wanted = _select_by_motion(meta, all_keys, cfg.min_dist)

    src_cap = cv2.VideoCapture(str(video_path))
    src_width = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_cap.release()

    scale_x = cfg.width / src_width
    scale_y = cfg.height / src_height
    do_resize = (cfg.width != src_width) or (cfg.height != src_height)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    cap = cv2.VideoCapture(str(video_path))
    images: list[UInt8[Tensor, "..."]] = []
    cam_rows: list[np.ndarray] = []

    scene_name = iphone_dir.parent.name
    frame_iter = tqdm(
        enumerate(all_keys),
        total=len(all_keys),
        desc=f"  {scene_name}",
        position=tqdm_position,
        leave=False,
    )
    for i, key in frame_iter:
        if i not in wanted:
            cap.grab()  # advance without decoding
            continue
        ret, bgr = cap.read()
        if not ret:
            print(f"  warning: video ended early at frame {i}/{len(all_keys)}")
            break

        if do_resize:
            bgr = cv2.resize(bgr, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", bgr, encode_params)
        if not ok:
            continue
        images.append(torch.tensor(buf, dtype=torch.uint8))

        frame_meta = meta[key]
        c2w = np.array(frame_meta["aligned_pose"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        w2c_3x4 = w2c[:3, :].astype(np.float32).flatten()

        K = np.array(frame_meta["intrinsic"], dtype=np.float32)
        fx = K[0, 0] * scale_x
        fy = K[1, 1] * scale_y
        cx = K[0, 2] * scale_x
        cy = K[1, 2] * scale_y
        intrin = np.array(
            [fx / cfg.width, fy / cfg.height, cx / cfg.width, cy / cfg.height,
             float(cfg.width), float(cfg.height)],
            dtype=np.float32,
        )
        cam_rows.append(np.concatenate([intrin, w2c_3x4]))

    cap.release()

    if not images:
        return None

    cameras = torch.tensor(np.stack(cam_rows), dtype=torch.float32)
    return images, cameras


# Worker state is passed via a module-level variable set by the pool initializer
# so workers don't need to receive the large config through imap args.
_worker_cfg: ConvertConfig | None = None


def _init_worker(cfg: ConvertConfig, lock=None) -> None:
    global _worker_cfg
    _worker_cfg = cfg
    if lock is not None:
        tqdm.set_lock(lock)


def _scene_worker(args: tuple[str, Path]) -> tuple[str, Path | None, int]:
    """Process one scene and write the result to a temp file.

    Returns (scene_id, temp_path, num_bytes). temp_path is None on failure.
    """
    scene_id, temp_dir = args
    assert _worker_cfg is not None, "_init_worker was not called"
    cfg = _worker_cfg
    iphone_dir = cfg.input_dir / "data" / scene_id / "iphone"

    identity = mp.current_process()._identity
    worker_position = identity[0] if identity else 1

    result = load_frames_and_cameras(iphone_dir, cfg, tqdm_position=worker_position)
    if result is None:
        return scene_id, None, 0

    images, cameras = result
    n = len(images)
    scene_bytes = sum(img.numel() for img in images)

    example = {
        "key": f"{scene_id}_iphone",
        "url": "",
        "timestamps": torch.arange(n, dtype=torch.int64),
        "cameras": cameras,
        "images": images,
        "depths": [],
    }
    temp_path = temp_dir / f"{scene_id}.torch"
    torch.save(example, temp_path)
    return scene_id, temp_path, scene_bytes


def process_stage(
    stage: Literal["train", "test"],
    cfg: ConvertConfig,
    max_scenes: int | None = None,
    num_workers: int = 1,
) -> None:
    scene_ids = get_scene_ids(cfg.input_dir, stage)
    if max_scenes is not None:
        scene_ids = scene_ids[:max_scenes]
    print(
        f"[{stage}] {len(scene_ids)} scenes | "
        f"resolution {cfg.width}x{cfg.height} | min_dist {cfg.min_dist} m | "
        f"{num_workers} worker(s)"
    )

    chunk_size = 0
    chunk_index = 0
    chunk: list[dict] = []

    def save_chunk() -> None:
        nonlocal chunk_size, chunk_index, chunk
        key = f"{chunk_index:0>6}"
        out_dir = cfg.output_dir / stage
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  saving chunk {key} ({chunk_size / 1e6:.1f} MB, {len(chunk)} scenes)")
        torch.save(chunk, out_dir / f"{key}.torch")
        chunk_size = 0
        chunk_index += 1
        chunk = []

    def consume(scene_id: str, temp_path: Path | None, scene_bytes: int) -> None:
        nonlocal chunk_size
        if temp_path is None:
            print(f"  skipping {scene_id}: missing rgb.mkv or pose_intrinsic_imu.json")
            return
        chunk.append(torch.load(temp_path))
        temp_path.unlink()
        chunk_size += scene_bytes
        if chunk_size >= TARGET_BYTES_PER_CHUNK:
            save_chunk()

    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        worker_args = [(sid, temp_dir) for sid in scene_ids]

        if num_workers > 1:
            tqdm.set_lock(mp.RLock())
            with mp.Pool(
                num_workers,
                initializer=_init_worker,
                initargs=(cfg, tqdm.get_lock()),
            ) as pool:
                for result in tqdm(
                    pool.imap(_scene_worker, worker_args),
                    total=len(scene_ids),
                    desc=stage,
                    position=0,
                ):
                    consume(*result)
        else:
            _init_worker(cfg)
            for args in tqdm(worker_args, desc=stage, position=0):
                consume(*_scene_worker(args))

    if chunk_size > 0:
        save_chunk()

    print(f"[{stage}] building index.json ...")
    index = {}
    stage_dir = cfg.output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    for chunk_path in tqdm(sorted(stage_dir.glob("*.torch")), desc="indexing"):
        for ex in torch.load(chunk_path):
            index[ex["key"]] = chunk_path.name
    with (stage_dir / "index.json").open("w") as fh:
        json.dump(index, fh)
    print(f"[{stage}] done — {len(index)} scenes indexed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw ScanNet++ to .torch chunks")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="ScanNet++ root (contains data/ and splits/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output root; chunks written to <output-dir>/<stage>/")
    parser.add_argument("--stages", nargs="+", choices=["train", "test"], default=["test"],
                        help="Splits to process (default: test)")
    parser.add_argument("--resolution", type=str, default="920x690",
                        help="Output resolution as WxH (default: 920x690)")
    parser.add_argument("--min-dist", type=float, default=0.05,
                        help="Minimum camera translation in metres between kept frames (default: 0.05)")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Cap scenes per stage (useful for testing)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Parallel worker processes (default: 1)")
    args = parser.parse_args()

    try:
        width, height = (int(x) for x in args.resolution.split("x"))
    except ValueError:
        parser.error(f"--resolution must be WxH (e.g. 920x690), got: {args.resolution}")

    cfg = ConvertConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        width=width,
        height=height,
        min_dist=args.min_dist,
    )
    for stage in args.stages:
        process_stage(stage, cfg, args.max_scenes, args.num_workers)
