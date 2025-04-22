#!/usr/bin/env python3
"""
Interactive Video Sampling Demo

This script provides a simple command-line interface for generating
echo videos with various conditioning options.

Example usage:
    # Generate a video with class_id conditioning
    python video_sampler.py --config ../lvdm/configs/default.yaml \
        --unet ../models/unet --vae ../models/vae \
        --conditioning ../samples/data/reference_frames \
        --output ../samples/output/demo \
        --conditioning_type class_id --class_id 3 \
        --sampling_mode diffusion --steps 64 \
        --frames 192 --format mp4,jpg

    # Generate a video with LVEF conditioning
    python video_sampler.py --config ../lvdm/configs/default.yaml \
        --unet ../models/unet --vae ../models/vae \
        --conditioning ../samples/data/reference_frames \
        --output ../samples/output/demo \
        --conditioning_type lvef --lvef 55 \
        --sampling_mode flow_matching --steps 32 \
        --frames 64 --format gif

    # Generate a video with view conditioning and guidance
    python video_sampler.py --config ../lvdm/configs/default.yaml \
        --unet ../models/unet --vae ../models/vae \
        --conditioning ../samples/data/reference_frames \
        --output ../samples/output/demo \
        --conditioning_type view --view_id 2 \
        --sampling_mode diffusion --steps 128 \
        --guidance_scale 3.0 --frames 96 --format avi
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import json
import uuid
import time
from einops import rearrange
import random

# Add Echo-Dream to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import diffusers
from echo.common import (
    pad_reshape,
    unpad_reshape,
    padf,
    unpadf,
    load_model,
    save_as_mp4,
    save_as_gif,
    save_as_avi,
    save_as_img,
    parse_formats,
    FlowMatchingScheduler,
)
from echo.common.datasets import TensorSet, ImageSet, TensorSetv2
from torch.utils.data import DataLoader
from echo.lvdm.sample import get_conditioning_vector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Echo-Dream Video Sampler Demo")

    # Model paths
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file."
    )
    parser.add_argument("--unet", type=str, required=True, help="Path to UNet model.")
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE model.")
    parser.add_argument(
        "--conditioning",
        type=str,
        required=True,
        help="Path to conditioning frames or latent files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory for generated videos.",
    )

    # Conditioning parameters
    parser.add_argument(
        "--conditioning_type",
        type=str,
        default="class_id",
        choices=["class_id", "lvef", "view"],
        help="Type of conditioning to use.",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=None,
        help="Class ID to condition on (if conditioning_type='class_id').",
    )
    parser.add_argument(
        "--lvef",
        type=float,
        default=None,
        help="LVEF value to condition on (if conditioning_type='lvef').",
    )
    parser.add_argument(
        "--view_id",
        type=int,
        default=None,
        help="View ID to condition on (if conditioning_type='view').",
    )

    # Conditioning frame
    parser.add_argument(
        "--conditioning_frame",
        type=str,
        default=None,
        help="Specific conditioning frame to use (filename). If not specified, a random one will be chosen.",
    )

    # Sampling parameters
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="diffusion",
        choices=["diffusion", "flow_matching"],
        help="Sampling method to use.",
    )
    parser.add_argument(
        "--steps", type=int, default=64, help="Number of sampling steps."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (if > 1.0).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=192,
        help="Number of frames to generate. Must be a multiple of 32.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )

    # Output parameters
    parser.add_argument(
        "--format",
        type=parse_formats,
        default=["mp4"],
        help="Save formats separated by commas (e.g., mp4,jpg). Available: avi, mp4, gif, jpg, png, pt",
    )

    args = parser.parse_args()

    # Validate conditioning parameters
    if args.conditioning_type == "class_id" and args.class_id is None:
        args.class_id = random.randint(0, 9)
        print(f"No class_id specified, using random class_id: {args.class_id}")

    if args.conditioning_type == "lvef" and args.lvef is None:
        args.lvef = random.uniform(20, 80)
        print(f"No lvef specified, using random lvef: {args.lvef:.1f}")

    if args.conditioning_type == "view" and args.view_id is None:
        args.view_id = random.randint(0, 3)
        print(f"No view_id specified, using random view_id: {args.view_id}")

    return args


def setup_models_and_scheduler(args):
    """Set up models and scheduler."""
    print(f"Loading configuration from {args.config}")
    config = OmegaConf.load(args.config)

    print(f"Loading UNet from {args.unet}")
    unet = load_model(args.unet)

    print(f"Loading VAE from {args.vae}")
    vae = load_model(args.vae)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    unet = unet.to(device)
    vae = vae.to(device)

    # Set up scheduler based on sampling mode
    if args.sampling_mode == "diffusion":
        print("Setting up diffusion scheduler")
        scheduler_kwargs = OmegaConf.to_container(config.noise_scheduler)
        scheduler_klass_name = scheduler_kwargs.pop("_class_name")
        scheduler_klass = getattr(diffusers, scheduler_klass_name, None)
        assert scheduler_klass is not None, (
            f"Could not find scheduler class {scheduler_klass_name}"
        )
        scheduler = scheduler_klass(**scheduler_kwargs)
    else:  # flow_matching
        print("Setting up flow matching scheduler")
        scheduler = FlowMatchingScheduler(
            num_train_timesteps=config.get("num_train_timesteps", 1000)
        )

    scheduler.set_timesteps(args.steps)

    return config, unet, vae, scheduler, device


def load_conditioning_frames(args):
    """Load conditioning frames or latents."""
    print(f"Loading conditioning data from {args.conditioning}")

    # Check file extension
    files = os.listdir(args.conditioning)
    if not files:
        raise ValueError(f"No files found in {args.conditioning}")

    # Get the file extension
    file_ext = files[0].split(".")[-1].lower()
    if file_ext not in ["pt", "jpg", "png"]:
        raise ValueError(
            f"Unsupported file extension: {file_ext}. Must be pt, jpg, or png."
        )

    # Load dataset based on extension
    if file_ext == "pt":
        dataset = TensorSetv2(args.conditioning)
        print(f"Loaded {len(dataset)} tensor files")
    else:
        dataset = ImageSet(args.conditioning, ext=file_ext)
        print(f"Loaded {len(dataset)} {file_ext} images")

    # Select specific file or random file
    if args.conditioning_frame and args.conditioning_frame in files:
        frame_path = os.path.join(args.conditioning, args.conditioning_frame)
        print(f"Using specified conditioning frame: {args.conditioning_frame}")
    else:
        if args.conditioning_frame:
            print(
                f"Warning: Specified conditioning frame {args.conditioning_frame} not found. Using random frame."
            )

        # Use a random frame
        frame_idx = random.randint(0, len(dataset) - 1)
        frame_path = dataset.files[frame_idx]
        frame_name = os.path.basename(frame_path)
        print(f"Selected random conditioning frame: {frame_name}")

    return dataset, frame_path, file_ext


def generate_video(args, config, unet, vae, scheduler, device, frame_path, file_ext):
    """Generate video using the specified parameters."""
    print("Starting video generation...")

    # Setup seed
    if args.seed is None:
        args.seed = int(time.time()) % 1000000
    print(f"Using seed: {args.seed}")

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Determine format functions based on UNet type
    format_input = (
        pad_reshape
        if config.unet._class_name == "UNetSpatioTemporalConditionModel"
        else padf
    )
    format_output = (
        unpad_reshape
        if config.unet._class_name == "UNetSpatioTemporalConditionModel"
        else unpadf
    )

    # Setup dimensions
    B = 1  # Just generate one video
    C = config.unet.out_channels
    T = config.unet.num_frames
    H = W = config.unet.sample_size

    # Stitching parameters to handle longer videos
    args.frames = int(
        np.ceil(args.frames / 32) * 32
    )  # Make sure frames are a multiple of 32
    if args.frames > T:
        OT = T // 2  # overlap
        TR = (args.frames - T) / (T - OT) + 1
        TR = int(np.ceil(TR))  # ceiling
        NT = args.frames
    else:
        OT = 0
        TR = 1
        NT = T

    print(f"Video dimensions: Frames={NT}, Height={H}, Width={W}")
    print(f"Stitching parameters: Base frames={T}, Overlap={OT}, Repetitions={TR}")

    # Load conditioning frame
    if file_ext == "pt":
        latent_cond_images = torch.load(frame_path).to(device)
        if latent_cond_images.dim() == 5:  # B, C, C, H, W (TensorSetv2 format)
            latent_cond_images = latent_cond_images.squeeze(1)
    else:  # jpg, png
        from torchvision import transforms
        from PIL import Image

        # Load image
        image = Image.open(frame_path).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((H, W)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Project to latent space
        with torch.no_grad():
            latent_cond_images = (
                vae.encode(image_tensor).latent_dist.sample()
                * vae.config.scaling_factor
            )

    # Ensure proper shape
    if latent_cond_images.dim() == 3:  # C, H, W
        latent_cond_images = latent_cond_images.unsqueeze(0)  # B, C, H, W

    # Generate noise
    latents = torch.randn((B, C, NT, H, W), device=device, generator=generator)

    # Get conditioning value based on type
    if args.conditioning_type == "class_id":
        conditioning_value = args.class_id
    elif args.conditioning_type == "lvef":
        conditioning_value = args.lvef
    else:  # view
        conditioning_value = args.view_id

    # Create conditioning vector
    conditioning = get_conditioning_vector(
        args.conditioning_type, conditioning_value, B, device, torch.float32
    )

    # Repeat conditioning for temporal stitching if needed
    conditioning = conditioning.repeat_interleave(TR, dim=0) if TR > 1 else conditioning

    # Expand conditioning frame to video
    latent_cond_images = latent_cond_images[:, :, None, :, :].repeat(1, 1, NT, 1, 1)

    # Forward kwargs setup
    forward_kwargs = {"timestep": -1}  # Will be updated in the loop

    if args.conditioning_type == "text":
        forward_kwargs["encoder_hidden_states"] = conditioning
    else:
        forward_kwargs["encoder_hidden_states"] = conditioning

    if config.unet._class_name == "UNetSpatioTemporalConditionModel":
        dummy_added_time_ids = torch.zeros(
            (B * TR, config.unet.addition_time_embed_dim),
            device=device,
            dtype=torch.float32,
        )
        forward_kwargs["added_time_ids"] = dummy_added_time_ids

    # Set up progress bar
    timesteps = scheduler.timesteps
    progress_bar = tqdm(timesteps, desc="Generating video")

    # Denoising loop
    with torch.no_grad():
        for t in progress_bar:
            forward_kwargs["timestep"] = t

            # Prepare model input
            latent_model_input = scheduler.scale_model_input(latents, timestep=t)
            latent_model_input = torch.cat(
                (latent_model_input, latent_cond_images), dim=1
            )

            # Handle classifier-free guidance
            use_guidance = args.guidance_scale > 1.0

            if use_guidance and args.sampling_mode == "diffusion":
                # Create unconditional input
                uncond_kwargs = forward_kwargs.copy()
                uncond_kwargs["encoder_hidden_states"] = torch.zeros_like(conditioning)

                # Format inputs
                latent_model_input, padding = format_input(latent_model_input, mult=3)

                # Stitching for conditional prediction
                inputs = torch.cat(
                    [
                        latent_model_input[:, r * (T - OT) : r * (T - OT) + T]
                        for r in range(TR)
                    ],
                    dim=0,
                )

                # Conditional and unconditional predictions
                noise_pred_cond = unet(inputs, **forward_kwargs).sample
                noise_pred_uncond = unet(inputs, **uncond_kwargs).sample

                # Apply guidance
                outputs_cond = torch.chunk(noise_pred_cond, TR, dim=0)
                outputs_uncond = torch.chunk(noise_pred_uncond, TR, dim=0)

                noise_predictions = []
                for r in range(TR):
                    cond_chunk = outputs_cond[r] if r == 0 else outputs_cond[r][:, OT:]
                    uncond_chunk = (
                        outputs_uncond[r] if r == 0 else outputs_uncond[r][:, OT:]
                    )
                    guided_chunk = uncond_chunk + args.guidance_scale * (
                        cond_chunk - uncond_chunk
                    )
                    noise_predictions.append(guided_chunk)

                noise_pred = torch.cat(noise_predictions, dim=1)
            else:
                # Standard prediction without guidance
                latent_model_input, padding = format_input(latent_model_input, mult=3)

                inputs = torch.cat(
                    [
                        latent_model_input[:, r * (T - OT) : r * (T - OT) + T]
                        for r in range(TR)
                    ],
                    dim=0,
                )

                noise_pred = unet(inputs, **forward_kwargs).sample
                outputs = torch.chunk(noise_pred, TR, dim=0)

                noise_predictions = []
                for r in range(TR):
                    noise_predictions.append(
                        outputs[r] if r == 0 else outputs[r][:, OT:]
                    )

                noise_pred = torch.cat(noise_predictions, dim=1)

            noise_pred = format_output(noise_pred, pad=padding)

            # Update latents
            if args.sampling_mode == "diffusion":
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            else:  # flow_matching
                dt = 1.0 / (len(timesteps) - 1)
                latents = latents - noise_pred * dt

    print("Decoding with VAE...")

    # Decode with VAE
    latents = latents / vae.config.scaling_factor

    # Process in chunks to save memory
    latents = rearrange(latents, "b c t h w -> (b t) c h w")
    chunk_size = 16  # Process 16 frames at a time
    video_chunks = []

    for i in range(0, latents.shape[0], chunk_size):
        chunk = latents[i : i + chunk_size].to(device)
        with torch.no_grad():
            video_chunk = vae.decode(chunk).sample
        video_chunks.append(video_chunk.cpu())

    video = torch.cat(video_chunks, dim=0)  # (B*T) x C x H x W

    # Format output
    video = rearrange(video, "(b t) c h w -> b t h w c", b=B)
    video = (video + 1) * 128
    video = video.clamp(0, 255).to(torch.uint8)[0]  # Remove batch dimension

    print(f"Video generated with shape: {video.shape}")

    return video


def save_video(video, args):
    """Save the generated video in the requested formats."""
    # Create output directories
    os.makedirs(args.output, exist_ok=True)

    for fmt in args.format:
        os.makedirs(os.path.join(args.output, fmt), exist_ok=True)

    # Generate a unique ID for this video
    video_id = f"video_{uuid.uuid4().hex[:8]}"

    # Save metadata
    metadata = {
        "conditioning_type": args.conditioning_type,
        "sampling_mode": args.sampling_mode,
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
        "frames": args.frames,
        "seed": args.seed,
        "dimensions": {
            "height": video.shape[1],
            "width": video.shape[2],
            "frames": video.shape[0],
        },
    }

    # Add specific conditioning value to metadata
    if args.conditioning_type == "class_id":
        metadata["class_id"] = args.class_id
    elif args.conditioning_type == "lvef":
        metadata["lvef"] = args.lvef
    else:  # view
        metadata["view_id"] = args.view_id

    # Save in requested formats
    saved_paths = []

    if "mp4" in args.format:
        mp4_path = os.path.join(args.output, "mp4", f"{video_id}.mp4")
        save_as_mp4(video, mp4_path)
        saved_paths.append(mp4_path)

    if "avi" in args.format:
        avi_path = os.path.join(args.output, "avi", f"{video_id}.avi")
        save_as_avi(video, avi_path)
        saved_paths.append(avi_path)

    if "gif" in args.format:
        gif_path = os.path.join(args.output, "gif", f"{video_id}.gif")
        save_as_gif(video, gif_path)
        saved_paths.append(gif_path)

    if "jpg" in args.format:
        jpg_dir = os.path.join(args.output, "jpg", video_id)
        save_as_img(video, jpg_dir, ext="jpg")
        saved_paths.append(jpg_dir)

    # Save metadata
    metadata_path = os.path.join(args.output, f"{video_id}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nVideo saved to:")
    for path in saved_paths:
        print(f"  - {path}")
    print(f"  - {metadata_path} (metadata)")

    return video_id, saved_paths


def main():
    """Main function."""
    args = parse_args()

    # Setup models and scheduler
    config, unet, vae, scheduler, device = setup_models_and_scheduler(args)

    # Load conditioning frames
    dataset, frame_path, file_ext = load_conditioning_frames(args)

    # Generate video
    video = generate_video(
        args, config, unet, vae, scheduler, device, frame_path, file_ext
    )

    # Save video
    video_id, saved_paths = save_video(video, args)

    print(
        f"\nDone! Generated video {video_id} with {args.conditioning_type}={getattr(args, args.conditioning_type)}"
    )


if __name__ == "__main__":
    main()
