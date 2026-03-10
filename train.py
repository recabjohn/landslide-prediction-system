"""
YOLOv8 Training Script for Landslide Detection
=================================================
Trains a YOLOv8 object detection model on satellite imagery
annotated with landslide, debris_flow, and normal_terrain classes.

Usage:
    python train.py --data dataset.yaml --model yolov8n.pt --epochs 50 --imgsz 640

    Or with all options:
    python train.py \\
        --data dataset.yaml \\
        --model yolov8n.pt \\
        --epochs 100 \\
        --imgsz 640 \\
        --batch 16 \\
        --device 0 \\
        --project runs/landslide \\
        --name experiment_1

Equivalent YOLO CLI command:
    yolo detect train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
"""

import argparse
import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for landslide detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training with nano model
  python train.py --data dataset.yaml --model yolov8n.pt --epochs 50

  # Full training with medium model on GPU
  python train.py --data dataset.yaml --model yolov8m.pt --epochs 100 --batch 16 --device 0

  # Resume training from a checkpoint
  python train.py --resume runs/landslide/experiment_1/weights/last.pt
        """,
    )

    # Required arguments
    parser.add_argument("--data", type=str, default="dataset.yaml",
                        help="Path to dataset YAML configuration (default: dataset.yaml)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Pre-trained model weights to start from (default: yolov8n.pt)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size in pixels (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate (default: 0.01)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final learning rate fraction (default: 0.01)")

    # Augmentation
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable data augmentation (default: True)")
    parser.add_argument("--mosaic", type=float, default=1.0,
                        help="Mosaic augmentation probability (default: 1.0)")
    parser.add_argument("--flipud", type=float, default=0.5,
                        help="Vertical flip probability (default: 0.5)")
    parser.add_argument("--fliplr", type=float, default=0.5,
                        help="Horizontal flip probability (default: 0.5)")

    # Hardware and output
    parser.add_argument("--device", type=str, default="",
                        help="Device: 'cpu', '0', '0,1', etc. (default: auto)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of data loading workers (default: 8)")
    parser.add_argument("--project", type=str, default="runs/landslide",
                        help="Output project directory (default: runs/landslide)")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (default: auto-generated)")

    # Training modes
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a checkpoint path")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience in epochs (default: 20)")
    parser.add_argument("--save-period", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")

    # Validation
    parser.add_argument("--val", action="store_true", default=True,
                        help="Run validation during training (default: True)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for validation (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="IoU threshold for NMS (default: 0.7)")

    return parser.parse_args()


def validate_dataset(data_path):
    """
    Validate that the dataset structure exists and contains images.

    Args:
        data_path (str): Path to dataset.yaml

    Returns:
        bool: True if dataset is valid
    """
    import yaml

    if not os.path.isfile(data_path):
        logger.error(f"Dataset config not found: {data_path}")
        return False

    with open(data_path, "r") as f:
        config = yaml.safe_load(f)

    root = config.get("path", ".")

    for split in ["train", "val"]:
        split_path = os.path.join(root, config.get(split, f"{split}/images"))
        if os.path.isdir(split_path):
            images = [f for f in os.listdir(split_path)
                      if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))]
            logger.info(f"  {split}: {len(images)} images found in {split_path}")
            if len(images) == 0:
                logger.warning(f"  ⚠ No images found in {split_path}")
        else:
            logger.warning(f"  ⚠ Split directory not found: {split_path}")

    return True


def train(args):
    """
    Execute YOLOv8 training.

    Args:
        args: Parsed command-line arguments
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "The 'ultralytics' package is not installed.\n"
            "Install it with: pip install ultralytics"
        )
        sys.exit(1)

    # Generate experiment name if not provided
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"landslide_{timestamp}"

    logger.info("=" * 60)
    logger.info("  LANDSLIDE DETECTION - YOLOv8 TRAINING")
    logger.info("=" * 60)
    logger.info(f"  Dataset     : {args.data}")
    logger.info(f"  Base Model  : {args.model}")
    logger.info(f"  Epochs      : {args.epochs}")
    logger.info(f"  Image Size  : {args.imgsz}")
    logger.info(f"  Batch Size  : {args.batch}")
    logger.info(f"  Device      : {args.device or 'auto'}")
    logger.info(f"  Experiment  : {args.project}/{args.name}")
    logger.info("=" * 60)

    # Validate dataset
    logger.info("\nValidating dataset...")
    validate_dataset(args.data)

    # Handle resume
    if args.resume:
        logger.info(f"\nResuming training from: {args.resume}")
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        # Load pre-trained model
        logger.info(f"\nLoading base model: {args.model}")
        model = YOLO(args.model)

        # Start training
        logger.info("\nStarting training...\n")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            lrf=args.lrf,
            device=args.device if args.device else None,
            workers=args.workers,
            project=args.project,
            name=args.name,
            patience=args.patience,
            save_period=args.save_period,
            val=args.val,
            conf=args.conf,
            iou=args.iou,
            augment=args.augment,
            mosaic=args.mosaic,
            flipud=args.flipud,
            fliplr=args.fliplr,
            verbose=True,
        )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)

    # Best model weights path
    best_weights = os.path.join(args.project, args.name, "weights", "best.pt")
    last_weights = os.path.join(args.project, args.name, "weights", "last.pt")

    if os.path.isfile(best_weights):
        logger.info(f"  Best weights : {best_weights}")
    if os.path.isfile(last_weights):
        logger.info(f"  Last weights : {last_weights}")

    logger.info(f"\n  To use the trained model:")
    logger.info(f"    from ultralytics import YOLO")
    logger.info(f"    model = YOLO('{best_weights}')")
    logger.info(f"    results = model('satellite_image.jpg')")
    logger.info("=" * 60)

    # Run validation on test set if available
    logger.info("\nRunning final validation...")
    try:
        metrics = model.val()
        logger.info(f"  mAP50     : {metrics.box.map50:.4f}")
        logger.info(f"  mAP50-95  : {metrics.box.map:.4f}")
    except Exception as e:
        logger.warning(f"  Validation skipped: {e}")

    return results


def main():
    """Main entry point."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
