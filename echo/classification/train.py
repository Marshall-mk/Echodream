import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import set_seed
import warnings

warnings.filterwarnings("ignore")

from echo.classification.model import get_video_classifier
from echo.classification.data import create_video_dataloaders
import imageio
from echo.common import save_as_mp4


def train_epoch(model, dataloader, criterion, optimizer, accelerator, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    debug = False
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if debug == True and batch_idx == 0 and accelerator.is_local_main_process:
            debug_dir = "debug_batches/"
            os.makedirs(debug_dir, exist_ok=True)
            for i, video in enumerate(inputs):
                video_path = (
                    debug_dir
                    + f"class_{targets[i].item()}_batch_{batch_idx}_video_{i}.mp4"
                )
                video_np = video.cpu().numpy()
                # Permute dimensions to (T, H, W, C)
                video_np = np.transpose(video_np, (0, 2, 3, 1))
                # Scale to 0-255 and convert to uint8
                video_np = (video_np * 255).astype(np.uint8)
                with imageio.get_writer(
                    video_path,
                    fps=32,
                ) as writer:
                    for i in range(video_np.shape[0]):
                        writer.append_data(video_np[i])

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        accelerator.backward(loss)
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0 and accelerator.is_local_main_process:
            print(
                f"Epoch: {epoch} | Batch: {batch_idx}/{len(dataloader)} | "
                f"Loss: {running_loss / (batch_idx + 1):.4f} | "
                f"Acc: {100.0 * correct / total:.2f}%"
            )

    # Convert metrics to tensors before gathering
    loss_tensor = torch.tensor(
        running_loss / len(dataloader), device=accelerator.device
    )
    acc_tensor = torch.tensor(100.0 * correct / total, device=accelerator.device)

    # Gather and reduce metrics from all processes
    loss_tensor = accelerator.gather(loss_tensor).mean()
    acc_tensor = accelerator.gather(acc_tensor).mean()

    return loss_tensor.item(), acc_tensor.item()


def validate(model, dataloader, criterion, accelerator):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Convert metrics to tensors before gathering
    loss_tensor = torch.tensor(
        running_loss / len(dataloader), device=accelerator.device
    )
    acc_tensor = torch.tensor(100.0 * correct / total, device=accelerator.device)

    # Gather and reduce metrics from all processes
    loss_tensor = accelerator.gather(loss_tensor).mean()
    acc_tensor = accelerator.gather(acc_tensor).mean()

    if accelerator.is_local_main_process:
        print(
            f"Validation Loss: {loss_tensor.item():.4f} | Validation Acc: {acc_tensor.item():.2f}%"
        )

    return loss_tensor.item(), acc_tensor.item()


def test(model, dataloader, accelerator):
    """Test the model"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)

            _, predicted = outputs.max(1)

            # Move to CPU for collection
            pred_cpu = accelerator.gather(predicted).cpu().numpy()
            targets_cpu = accelerator.gather(targets).cpu().numpy()

            all_preds.extend(pred_cpu)
            all_targets.extend(targets_cpu)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Convert metric to tensor before gathering
    acc_tensor = torch.tensor(100.0 * correct / total, device=accelerator.device)

    # Gather and reduce metrics from all processes
    acc_tensor = accelerator.gather(acc_tensor).mean()

    if accelerator.is_local_main_process:
        print(f"Test Acc: {acc_tensor.item():.2f}%")

    return acc_tensor.item(), all_preds, all_targets


def parse_args():
    parser = argparse.ArgumentParser(description="Video Classification Training")

    # Data paths
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV file with video metadata",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing the video frames",
    )
    parser.add_argument(
        "--synthetic-csv-path",
        type=str,
        default=None,
        help="Path to CSV file with synthetic video metadata",
    )
    parser.add_argument(
        "--synthetic-data-dir",
        type=str,
        default=None,
        help="Directory containing synthetic video frames",
    )
    parser.add_argument(
        "--use-synthetic-for",
        type=str,
        nargs="+",
        default=[],
        choices=["train", "val", "test"],
        help="Which splits should use synthetic data",
    )

    # Model parameters
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "r3d_18",
            "r2plus1d_18",
        ],
        help="Model backbone to use",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained weights"
    )
    parser.add_argument(
        "--pool-type",
        type=str,
        default="avg",
        choices=["avg", "max", "attention"],
        help="Type of temporal pooling",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=45, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Initial learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=64,
        help="Number of frames to sample per video",
    )
    parser.add_argument(
        "--frame-sampling",
        type=str,
        default="uniform",
        choices=["uniform", "random"],
        help="Method for sampling frames",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/classification/real",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="Name of experiment"
    )

    # New arguments for accelerate
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=False,
        help="Enable pin_memory for dataloaders (disable if CUDA errors occur)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Set up output directory
    if args.experiment_name is None:
        args.experiment_name = f"{args.backbone}_{int(time.time())}"

    output_dir = Path(args.output_dir) / args.experiment_name

    # Only create directories and write files from the main process
    if accelerator.is_local_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save args
        with open(output_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        # Set up tensorboard
        writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    else:
        writer = None

    # Print info about distributed setup
    if accelerator.is_local_main_process:
        print(f"Using device: {accelerator.device}")
        print(f"Num processes: {accelerator.num_processes}")
        print(f"Distributed type: {accelerator.distributed_type}")

    # Set up dataloaders
    if accelerator.is_local_main_process:
        print("Creating dataloaders...")

    dataloader_config = {
        "csv_path": args.csv_path,
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "frames_per_clip": args.frames_per_clip,
        "frame_sampling": args.frame_sampling,
        "synthetic_csv_path": args.synthetic_csv_path,
        "synthetic_data_dir": args.synthetic_data_dir,
        "use_synthetic_for": args.use_synthetic_for,
        "pin_memory": args.pin_memory,  # Add pin_memory control
    }

    # Add debug message for pin_memory setting
    if accelerator.is_local_main_process:
        print(f"Using pin_memory: {args.pin_memory}")

    dataloaders = create_video_dataloaders(dataloader_config)

    # Get number of classes
    num_classes = len(dataloaders["train"].dataset.classes)
    if accelerator.is_local_main_process:
        print(f"Number of classes: {num_classes}")

    # Create model
    if accelerator.is_local_main_process:
        print(f"Creating model: {args.backbone}")

    # Prepare model kwargs based on model type
    model_kwargs = {"num_classes": num_classes}

    if args.backbone in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        # model_kwargs['model_name'] = args.backbone
        model_kwargs["pretrained"] = args.pretrained
        model_kwargs["pool_type"] = args.pool_type
    elif args.backbone in ["r3d_18", "r2plus1d_18"]:
        # model_kwargs['model_name'] = args.backbone
        model_kwargs["pretrained"] = args.pretrained
        model_kwargs["dropout_prob"] = 0.5
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    # Pass the backbone as a parameter, not as part of kwargs
    # This ensures the correct code path in get_video_classifier is taken
    model = get_video_classifier(
        num_classes=model_kwargs.pop("num_classes"),
        backbone=args.backbone,
        **model_kwargs,
    )
    print(f"Class weights: {dataloaders['train'].dataset.class_weights}")
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss(
        weight=dataloaders["train"].dataset.class_weights.to(accelerator.device)
    )
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Prepare for distributed training with accelerate
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, dataloaders["train"], dataloaders["val"]
    )
    test_dataloader = accelerator.prepare(dataloaders["test"])

    # Train model
    if accelerator.is_local_main_process:
        print("Starting training...")

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_dataloader, criterion, optimizer, accelerator, epoch
        )

        val_loss, val_acc = validate(model, val_dataloader, criterion, accelerator)

        # Update learning rate
        scheduler.step()

        # Log metrics (only on main process)
        if accelerator.is_local_main_process and writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        # Save best model (only on main process)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            if accelerator.is_local_main_process:
                # Unwrap model before saving
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "classes": dataloaders["train"].dataset.classes,
                    },
                    output_dir / "best_model.pth",
                )
                print(f"Model saved at epoch {epoch} with val acc: {val_acc:.2f}%")

        # Save latest model (only on main process)
        if accelerator.is_local_main_process:
            # Unwrap model before saving
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "classes": dataloaders["train"].dataset.classes,
                },
                output_dir / "latest_model.pth",
            )

    # Make sure all processes are synced before testing
    accelerator.wait_for_everyone()

    # Test best model
    if accelerator.is_local_main_process:
        print("Loading best model for testing...")

    checkpoint = torch.load(
        output_dir / "best_model.pth", map_location=accelerator.device
    )
    # Unwrap the model for loading weights
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint["model_state_dict"])

    test_acc, all_preds, all_targets = test(model, test_dataloader, accelerator)

    # Save test results (only on main process)
    if accelerator.is_local_main_process:
        if writer is not None:
            writer.add_scalar("Accuracy/test", test_acc, 0)

        np.savez(
            output_dir / "test_results.npz",
            predictions=all_preds,
            targets=all_targets,
            accuracy=test_acc,
        )

        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Test accuracy: {test_acc:.2f}%")

        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
