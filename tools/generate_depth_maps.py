"""
Generate depth maps for NuScenes images using Depth Anything 3 (DA3).

Prerequisites:
    Install depth-anything-3 package:
        cd Depth-Anything-3 && pip install -e .

    The DA3NESTED-GIANT-LARGE-1.1 model will be auto-downloaded from
    Hugging Face on first run, or you can pre-download it:
        huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir ckpts/da3

Usage:
    python tools/generate_depth_maps.py --dataroot data/nuScenes --output data/DA3
    python tools/generate_depth_maps.py --dataroot data/nuScenes --output data/DA3 --version v1.0-mini
    python tools/generate_depth_maps.py --model-name /path/to/local/da3
    python tools/generate_depth_maps.py --process-res 504 --resume
    python -m torch.distributed.run --nproc_per_node=4 tools/generate_depth_maps.py --dataroot data/nuScenes --output data/DA3 --resume
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from depth_anything_3.api import DepthAnything3


CAM_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate depth maps using DA3")
    parser.add_argument(
        "--dataroot",
        type=str,
        default="data/nuScenes",
        help="NuScenes data root directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/DA3",
        help="Output directory for depth maps",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="NuScenes dataset version",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="DA3 model name (HuggingFace hub ID or local path)",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Processing resolution for DA3 inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images per DA3 inference call",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed images",
    )
    return parser.parse_args()


def collect_image_paths(nusc: NuScenes):
    """Collect all unique camera image sample_data entries from NuScenes."""
    image_entries = []
    seen_tokens = set()

    for sample in nusc.sample:
        for cam in CAM_CHANNELS:
            sd_token = sample["data"][cam]
            if sd_token not in seen_tokens:
                seen_tokens.add(sd_token)
                sd = nusc.get("sample_data", sd_token)
                image_entries.append(sd)

    return image_entries


def get_output_path(output_dir: str, filename: str) -> str:
    """Convert NuScenes filename to depth map output path.

    Input:  samples/CAM_FRONT/xxx.jpg
    Output: data/DA3/samples/CAM_FRONT/xxx.npz
    """
    mask_filename = Path(filename).with_suffix(".npz")
    return os.path.join(output_dir, str(mask_filename))


def setup_distributed():
    """Initialize distributed environment from torchrun env vars."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return False, rank, world_size, local_rank

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world_size, local_rank


def cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@torch.no_grad()
def process_batch(model, image_paths, process_res):
    """Process a batch of images through DA3 to get depth maps.

    Returns list of dicts, each containing:
        - depth: np.ndarray (H, W), float32 depth map
        - sky: np.ndarray (H, W), bool sky mask (if available)
    """
    prediction = model.inference(
        image=image_paths,
        process_res=process_res,
        export_dir=None,
    )

    results = []
    n = prediction.depth.shape[0]
    for i in range(n):
        entry = {"depth": prediction.depth[i]}
        if prediction.sky is not None:
            entry["sky"] = prediction.sky[i]
        if prediction.conf is not None:
            entry["conf"] = prediction.conf[i]
        results.append(entry)

    return results


def main():
    args = parse_args()

    is_distributed, rank, world_size, local_rank = setup_distributed()

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"

    if rank == 0:
        print(f"Using device: {device}")
        if is_distributed:
            print(f"Using DDP with world size: {world_size}")

    # Load DA3 model
    if rank == 0:
        print(f"Loading DA3 model: {args.model_name}")
    model = DepthAnything3.from_pretrained(args.model_name)
    model = model.to(device)
    if rank == 0:
        print("DA3 model loaded.")

    # Load NuScenes
    if rank == 0:
        print(f"Loading NuScenes {args.version} from {args.dataroot}")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    if rank == 0:
        print(f"Loaded {len(nusc.sample)} samples, {len(nusc.scene)} scenes.")

    # Collect all camera images
    image_entries = collect_image_paths(nusc)
    if rank == 0:
        print(f"Total camera images to process: {len(image_entries)}")

    # Filter already processed if resuming
    if args.resume:
        remaining = []
        for entry in image_entries:
            out_path = get_output_path(args.output, entry["filename"])
            if not os.path.exists(out_path):
                remaining.append(entry)
        if rank == 0:
            print(
                f"Resuming: {len(image_entries) - len(remaining)} already done, "
                f"{len(remaining)} remaining."
            )
        image_entries = remaining

    if len(image_entries) == 0:
        if rank == 0:
            print("All images already processed. Nothing to do.")
        cleanup_distributed(is_distributed)
        return

    total_images = len(image_entries)

    # Each rank processes a disjoint shard.
    local_entries = image_entries[rank::world_size]

    # Process in batches
    batch_size = args.batch_size
    local_num_batches = (len(local_entries) + batch_size - 1) // batch_size

    if is_distributed:
        max_batches_tensor = torch.tensor(local_num_batches, device=device)
        dist.all_reduce(max_batches_tensor, op=dist.ReduceOp.MAX)
        num_iters = int(max_batches_tensor.item())
    else:
        num_iters = local_num_batches

    pbar = tqdm(total=total_images, desc="Generating depth maps") if rank == 0 else None

    for batch_idx in range(num_iters):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(local_entries))
        batch_entries = local_entries[start:end]

        local_processed = 0

        if len(batch_entries) > 0:
            # Build image paths
            image_paths = [
                os.path.join(args.dataroot, entry["filename"])
                for entry in batch_entries
            ]

            # Run DA3 inference
            results = process_batch(model, image_paths, args.process_res)

            # Save depth maps
            for entry, result in zip(batch_entries, results):
                out_path = get_output_path(args.output, entry["filename"])
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.savez_compressed(out_path, **result)

            local_processed = len(batch_entries)

        if is_distributed:
            step_tensor = torch.tensor(local_processed, device=device)
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            global_processed = int(step_tensor.item())
        else:
            global_processed = local_processed

        if pbar is not None and global_processed > 0:
            pbar.update(global_processed)

    if pbar is not None:
        pbar.close()
        print(f"Done! Depth maps saved to {args.output}")

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
