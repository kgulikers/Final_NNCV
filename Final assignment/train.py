"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
import random
from argparse import ArgumentParser

import wandb
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize, AutoAugment
from torchvision.utils import make_grid

from unet import UNet

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    """
    Convert class IDs in the label image to train IDs.
    """
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

# convert train IDs to color images for visualization
def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    """
    Convert train IDs in the prediction tensor to a 3‑channel color image.
    Accepts `prediction` of shape [B, H, W] or [B, 1, H, W], returns [B, 3, H, W].
    """
    # ensure we have a channel dimension
    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(1)      # [B, 1, H, W]
    batch, _, height, width = prediction.shape

    # make sure we draw on the same device as the input
    device = prediction.device
    color_image = torch.zeros((batch, 3, height, width),
                              dtype=torch.uint8,
                              device=device)

    # for each train‑id, paint the matching pixels
    for train_id, col in train_id_to_color.items():
        mask = (prediction[:, 0] == train_id)      # [B, H, W] boolean
        for c in range(3):
            # in‑place fill; color[c] is an int 0–255
            color_image[:, c].masked_fill_(mask, col[c])

    return color_image


def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Path to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=9, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--img-height", type=int, default=1024, help="Height of input images")
    parser.add_argument("--img-width", type=int, default=1024, help="Width of input images")
    # Set very high so it doesnt unfreeze during training
    parser.add_argument("--freeze-epochs", type=int, default=5, help="Epochs with backbone frozen before unfreezing")
    return parser

def dice_loss(pred, target, smooth=1e-3):
    """
    Dice loss for semantic segmentation.
    """
    # Shape of the prediction tensor
    num_classes = pred.shape[1]

    # Don't use ignore_index for calculating weights
    target_clean = target.clone()
    target_clean[target_clean == 255] = 0
    target_onehot = F.one_hot(
        target_clean.long(), num_classes=num_classes
    ).permute(0, 3, 1, 2).float()

    # Use softmax to get class probabilities
    probs = F.softmax(pred, dim=1)

    # Calculate intersection and union
    intersection = (probs * target_onehot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    # Calculate dice score and smooth it 
    dice_score = (2 * intersection + smooth) / (union + smooth)
    mean_dice = dice_score.mean()

    # Return dice loss 
    return 1.0 - mean_dice

def log_cosh_dice_loss(pred, target, smooth=1e-3):
    """
    Log-cosh dice loss for semantic segmentation.
    """
    return torch.log(torch.cosh(dice_loss(pred, target, smooth)))

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for semantic segmentation.
    """
    def __init__(self, gamma=2, weight=None, ignore_index=255,
                 adaptive=True, beta=0.9, min_weight=0.5):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.adaptive = adaptive
        self.beta = beta
        self.min_weight = min_weight

    def forward(self, logits, targets):
        """
        Compute the focal loss.
        """
        # choose between adaptive or fixed class weights
        weights = (
            self.adaptive_weights(logits, targets)
            if self.adaptive
            else self.weight
        )
         # per‑pixel cross‑entropy (no reduction yet)
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=weights,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        # Estimate the probability of the true class
        prob_true = torch.exp(-ce_loss)

        # Calculate the focal factor
        focal_factor = (1 - prob_true) ** self.gamma
        # Return the mean of focal loss 
        return (focal_factor * ce_loss).mean()

    def adaptive_weights(self, logits, targets):
        """
        Calculate weights per class based on frequency of classes in the batch.
        """
        num_classes = logits.shape[1]

        # Dont use ignore_index for calculating weights
        mask = targets != self.ignore_index

        # Count pixels per class
        counts = torch.bincount(
            targets[mask].flatten(),
            minlength=num_classes
        ).float()

        # Normalize counts for probabilities
        total = counts.sum()
        if total > 0:
            class_probs = counts / total
        else:
            class_probs = torch.full(
                (num_classes,),
                1.0 / num_classes,
                device=targets.device
            )

        # Avoid zero devision
        min_prob = self.min_weight / num_classes
        class_probs = class_probs.clamp(min=min_prob)

        # Inverse frequency
        inv_freq = 1.0 / (class_probs + 1e-5)

        # Normalize weights 
        norm_weights = inv_freq / inv_freq.max()
        norm_weights = norm_weights.clamp(min=self.min_weight)

        # Apply beta to the weights
        weights = self.beta * norm_weights + (1 - self.beta)

        return weights.detach().to(logits.device)

def combined_loss(pred, target, alpha=0.5):
    """
    Combines focal loss and log-cosh dice loss.
    """
    return alpha * FocalLoss()(pred, target) + (1-alpha) * log_cosh_dice_loss(pred, target)

def compute_iou(pred, target, num_classes=19, ignore_index=255):
    """
    Compute IoU (Intersection over Union) for each class.
    """
    # Initialize list to store iou values 
    ious = []
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    # Loop over classes and calculate iou
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        # Calculate intersection and union
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        # Compute iou
        iou = intersection / union if union > 0 else float('nan')
        ious.append(iou)

    return ious

def compute_dice(pred, target, smooth=1e-3, num_classes=19):
    """
    Compute the Dice coefficient across all classes.
    """
    # Apply softmax for class probalilities 
    probs = F.softmax(pred, dim=1)

    # One hot encoding for the predicted classes 
    pred_onehot = F.one_hot(pred.argmax(dim=1), num_classes=num_classes)
    pred_onehot = pred_onehot.permute(0, 3, 1, 2).float()

    target = target.clone()
    target[target == 255] = 0

    # One hot encoding from target classes
    target_onehot = F.one_hot(target.long(), num_classes=num_classes)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    # Compute intersectin and union 
    intersection = (probs * target_onehot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    # Calculate dice coefficient
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def train_one_epoch(model, dataloader, optimizer, device, scaler=None,
                    grad_clip=1.0, scheduler=None, global_step=0, epoch=0):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):

        labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
        images, labels = images.to(device), labels.to(device)

        labels = labels.long().squeeze(1)  # Remove channel dimension

        optimizer.zero_grad()
        if scaler:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                logits, _ = model(images)
                loss = combined_loss(logits, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(images)
            loss = combined_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler:
            # Update the learning rate with scheduler
            scheduler.step()

        # Log the data to wandb
        wandb.log({
            "train_loss": loss.item(),
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1,
        }, step=epoch * len(dataloader) + i)
        
        global_step += 1
        running_loss += loss.item()

    return running_loss / len(dataloader), global_step

def validate(model, dataloader, device):
    """
    Validate the model on the validation set.
    """
    model.eval()
    
    # Lists to store validation results
    losses, all_iou, all_dice = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            # Forward pass
            logits, _ = model(images)
            loss = combined_loss(logits, labels)
            losses.append(loss.item())

            # Predictions and metric calculations
            preds = logits.argmax(dim=1)
            all_iou.append(compute_iou(preds, labels))
            all_dice.append(compute_dice(logits, labels))

    # Compute average loss and metrics
    avg_loss = sum(losses) / len(losses)
    per_class = torch.tensor(all_iou).nanmean(dim=0).tolist()  # Average IoU per class
    mean_iou = torch.tensor(per_class).nanmean().item()        # Mean IoU over all classes
    mean_dice = sum(all_dice) / len(all_dice)

    return avg_loss, mean_iou, mean_dice, per_class

# — Main Entry Point —
def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # data transforms & loaders
    img_size = (args.img_height, args.img_width)

    # Define the transforms to apply to the data
    train_transform = Compose([
        ToImage(),
        Resize(img_size),
        ToDtype(torch.float32, scale=True),
        Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # Transform for validation data
    valid_transform = Compose([
        ToImage(),
        Resize(img_size),
        ToDtype(torch.float32, scale=True),
        Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # Apply transform to the data 
    train_dataset  = Cityscapes(
        args.data_dir, 
        split="train", mode="fine", 
        target_type="semantic", 
        transforms=train_transform
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine",
        target_type="semantic", 
        transforms=valid_transform
    )
    
    # Wrap the dataset for transforms
    train_dataset  = wrap_dataset_for_transforms_v2(train_dataset )
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset , 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers
    )

    # Define the model
    model = UNet(
        num_classes=19,     
        image_size=img_size, 
        freeze_backbone=True
    ).to(device)

    # optimizer & scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4
    )
    # Calculate total steps for the scheduler
    total_steps = len(train_loader) * args.epochs

    # Define the scheduler
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=args.lr,
        total_steps=total_steps, 
        pct_start=0.1
    )

    # Initialize parameters for training progress
    best_valid_loss = float('inf')
    
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    current_best_model_path = None
    epochs_no_improve = 0
    global_step = 0

    # training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Unfreeze encoder (backbone) after freeze_epochs
        if epoch == args.freeze_epochs:
            print("Unfreezing encoder (backbone)")
            model.unfreeze_backbone()
            # Reinitialize optimizer and scheduler
            optimizer = AdamW([
                {"params": model.encoder.parameters(), "lr": args.lr * 0.1},
                {"params": model.decoder.parameters(), "lr": args.lr}
            ], weight_decay=1e-4)
            steps_left = (args.epochs - epoch + 1) * len(train_loader)
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=args.lr,
                total_steps=steps_left, 
                pct_start=0.05
            )

        print(f"\nEpoch {epoch}/{args.epochs}")
        # Start training the model
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device,
            scaler=scaler, scheduler=scheduler,
            global_step=global_step, epoch=epoch
        )
        print(f"Train loss: {train_loss:.4f}")

        # Validate the model
        val_loss, mean_iou, mean_dice, per_class_iou = validate(model, valid_loader, device)
        print(f"Validation loss: {val_loss:.4f}, mIoU: {mean_iou:.4f}, mDice: {mean_dice:.4f}")

        # Log the data to wandb
        iou_logs = {f"iou/{i}": per_class_iou[i] for i in range(len(per_class_iou))}
        wandb.log({
            "train_loss": train_loss,
            "valid_loss": val_loss,
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            **iou_logs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch": epoch
        }, step=global_step)

        # Predictions visualization to W&B
        model.eval()
        with torch.no_grad():
            imgs, lbls = next(iter(valid_loader))
            lbls = convert_to_train_id(lbls)
            preds = model(imgs.to(device))[0].argmax(dim=1)
            pred_color = convert_train_id_to_color(preds)
            label_color = convert_train_id_to_color(lbls)
            grid_pred = make_grid(pred_color, nrow=8).permute(1,2,0).cpu().numpy()
            grid_lbl = make_grid(label_color, nrow=8).permute(1,2,0).cpu().numpy()
            wandb.log({
                "predictions": [wandb.Image(grid_pred, caption="Prediction")],
                "ground_truth": [wandb.Image(grid_lbl, caption="Ground Truth")],
                "epoch": epoch
            }, step=global_step)

        # save best model checkpoint
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            if current_best_model_path:
                os.remove(current_best_model_path)
            current_best_model_path = os.path.join(
                output_dir,
                f"best_model-epoch={epoch:04}-val_loss={val_loss:.4f}.pth"
            )
            torch.save(model.state_dict(), current_best_model_path)

    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={val_loss:04}.pth"
        )
    )
    wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)