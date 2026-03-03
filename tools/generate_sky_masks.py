"""
Generate sky masks for NuScenes images using SAM3 (Segment Anything 3).

Prerequisites:
    Download SAM3 model to local directory before running:
        # Option 1: huggingface-cli (recommended)
        huggingface-cli download facebook/sam3 --local-dir ckpts/sam3

        # Option 2: Python
        from transformers import Sam3Processor, Sam3Model
        Sam3Model.from_pretrained("facebook/sam3").save_pretrained("ckpts/sam3")
        Sam3Processor.from_pretrained("facebook/sam3").save_pretrained("ckpts/sam3")

Usage:
    python tools/generate_sky_masks.py --dataroot data/nuScenes --output data/SAM3
    python tools/generate_sky_masks.py --dataroot data/nuScenes --output data/SAM3 --batch-size 8
    python tools/generate_sky_masks.py --dataroot data/nuScenes --output data/SAM3 --version v1.0-mini
    python tools/generate_sky_masks.py --model-name /path/to/custom/sam3
    python -m torch.distributed.run --nproc_per_node=4 tools/generate_sky_masks.py --dataroot data/nuScenes --output data/SAM3 --model-name ckpts/sam3 --batch-size 4 --resume
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
from transformers import Sam3Processor, Sam3Model


CAM_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sky masks using SAM3")
    parser.add_argument(
        "--dataroot",
        type=str,
        default="data/nuScenes",
        help="NuScenes data root directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/SAM3",
        help="Output directory for sky masks",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="NuScenes dataset version",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for SAM3 inference",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ckpts/sam3",
        help="Local path to SAM3 model directory (download first, see docstring)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for mask selection",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask binarization threshold",
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
    """Convert NuScenes filename to sky mask output path.

    Input:  samples/CAM_FRONT/xxx.jpg
    Output: data/SAM3/samples/CAM_FRONT/xxx.png
    """
    # Replace extension to .png (masks are saved as uint8 PNG)
    mask_filename = Path(filename).with_suffix(".png")
    return os.path.join(output_dir, str(mask_filename))


def extract_sky_mask(results, image_size):
    """Extract sky mask from SAM3 segmentation results.

    If SAM3 finds a 'sky' segment, return it as a binary mask.
    Otherwise return an all-zeros mask.
    """
    masks = results.get("masks", None)
    if masks is not None and len(masks) > 0:
        # Combine all sky masks (there may be multiple segments)
        combined = torch.zeros(image_size, dtype=torch.uint8)
        for mask in masks:
            if isinstance(mask, torch.Tensor):
                combined = combined | mask.cpu().to(torch.uint8)
            else:
                combined = combined | torch.from_numpy(np.array(mask)).to(torch.uint8)
        return combined.numpy()
    return np.zeros(image_size, dtype=np.uint8)


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
def process_batch(
    processor,
    model,
    images,
    device,
    threshold=0.5,
    mask_threshold=0.5,
):
    """Process a batch of images through SAM3 to get sky masks.

    Returns list of numpy arrays (H, W), each being a binary sky mask.
    """
    original_sizes = [img.size[::-1] for img in images]  # (W,H) -> (H,W)

    inputs = processor(
        images=images,
        text=["sky"] * len(images),
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)

    # Post-process to get instance segmentation results
    results_list = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=original_sizes,
    )

    sky_masks = []
    for results, (h, w) in zip(results_list, original_sizes):
        mask = extract_sky_mask(results, (h, w))
        sky_masks.append(mask)

    return sky_masks


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

    # Validate model path
    model_path = Path(args.model_name)
    if model_path.exists() and model_path.is_dir():
        if rank == 0:
            print(f"Using local SAM3 model: {model_path.resolve()}")
    elif not model_path.exists() and "/" not in args.model_name:
        raise FileNotFoundError(
            f"Local model directory not found: {args.model_name}\n"
            "Please download the model first:\n"
            "  huggingface-cli download facebook/sam3 --local-dir ckpts/sam3"
        )

    # Load SAM3 model
    if rank == 0:
        print(f"Loading SAM3 model: {args.model_name}")
    processor = Sam3Processor.from_pretrained(args.model_name)
    model = Sam3Model.from_pretrained(args.model_name).to(device)
    model.eval()
    if rank == 0:
        print("SAM3 model loaded.")

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

    pbar = tqdm(total=len(image_entries), desc="Generating sky masks") if rank == 0 else None

    for batch_idx in range(num_iters):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(local_entries))
        batch_entries = local_entries[start:end]

        local_processed = 0

        if len(batch_entries) > 0:
            # Load images
            images = []
            for entry in batch_entries:
                img_path = os.path.join(args.dataroot, entry["filename"])
                img = Image.open(img_path).convert("RGB")
                images.append(img)

            # Run SAM3 inference
            sky_masks = process_batch(
                processor,
                model,
                images,
                device,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
            )

            # Save masks
            for entry, mask in zip(batch_entries, sky_masks):
                out_path = get_output_path(args.output, entry["filename"])
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                # Save as uint8 PNG: 0 = not sky, 255 = sky
                mask_img = Image.fromarray(mask * 255)
                mask_img.save(out_path)

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
        print(f"Done! Sky masks saved to {args.output}")

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
