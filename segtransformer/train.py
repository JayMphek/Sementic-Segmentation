import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import random
from dataloader import IDDAWDataset
from segformer import create_segformer_model
from config import Config


def get_args():
    parser = argparse.ArgumentParser(description='Train SegFormer on IDDAW data')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--img_width', type=int, default=512, help='Target width of images')
    parser.add_argument('--img_height', type=int, default=256, help='Target height of images')
    parser.add_argument('--data_root', type=str, default='../IDDAW', help='Root directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples to use')
    parser.add_argument('--model_size', type=str, default='b2', choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'], help='SegFormer model size')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to specific checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def create_optimizer(model, lr, weight_decay=0.01):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )
    return optimizer

def create_scheduler(optimizer, total_steps, min_lr=1e-6):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=min_lr
    )
    return scheduler


def calculate_metrics(pred, target, num_classes, ignore_index=255):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    
    correct = (pred == target).sum()
    total = len(target)
    accuracy = correct / (total + 1e-10)
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    mean_iou = np.mean(ious) if ious else 0.0
    
    return accuracy, mean_iou


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, total_epochs, scaler=None, gradient_clip=1.0, log_interval=10):
    model.train()
    
    total_loss = 0
    total_acc = 0
    total_iou = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    
    for batch_idx, (images, masks, _) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            acc, iou = calculate_metrics(pred, masks, outputs.size(1), ignore_index=255)
        
        total_loss += loss.item()
        total_acc += acc
        total_iou += iou
        
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}',
                'iou': f'{iou:.4f}',
                'lr': f'{get_lr(optimizer):.6f}'
            })
    
    pbar.close()
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_acc, avg_iou


def validate(model, dataloader, criterion, device, num_classes):
    model.eval()
    
    total_loss = 0
    total_acc = 0
    total_iou = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            pred = outputs.argmax(dim=1)
            acc, iou = calculate_metrics(pred, masks, num_classes, ignore_index=255)
            
            total_loss += loss.item()
            total_acc += acc
            total_iou += iou
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}',
                'iou': f'{iou:.4f}'
            })
        
        pbar.close()
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_acc, avg_iou


def save_checkpoint(model, optimizer, epoch, best_iou, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou
    }
    path = os.path.join(save_dir, filename)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(checkpoint_path, model, optimizer):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 1, 0.0
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_iou = checkpoint.get('best_iou', 0.0)
    
    print(f" Resumed from epoch {checkpoint['epoch']}")
    print(f"   Previous best IoU: {best_iou:.4f}")
    print(f"   Starting from epoch {start_epoch}")
    
    return start_epoch, best_iou


def main():
    args = get_args()
    
    print("=" * 60)
    print("SegFormer Training Configuration")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: ({args.img_width}, {args.img_height})")
    print(f"Max samples: {args.max_samples}")
    print(f"Model size: {args.model_size}")
    print(f"Device: {args.device}")
    print(f"Resume: {args.resume}")
    print("=" * 60)
    
    set_seed(args.seed)
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using Apple Silicon GPU")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU")
    
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    target_size = (args.img_width, args.img_height)
    
    train_dataset = IDDAWDataset(
        root_dir=args.data_root,
        split='train',
        target_size=target_size
    )
    val_dataset = IDDAWDataset(
        root_dir=args.data_root,
        split='val',
        target_size=target_size
    )
    
    train_samples = int(args.max_samples * 0.70)
    val_samples = int(args.max_samples * 0.15)
    if len(train_dataset) > train_samples:
        indices = random.sample(range(len(train_dataset)), train_samples)
        train_dataset = Subset(train_dataset, indices)
        print(f"✅ Limited training dataset to {train_samples} samples")
    
    if len(val_dataset) > val_samples:
        indices = random.sample(range(len(val_dataset)), val_samples)
        val_dataset = Subset(val_dataset, indices)
        print(f"✅ Limited validation dataset to {val_samples} samples")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    if isinstance(train_dataset, Subset):
        num_classes = train_dataset.dataset.num_classes
    else:
        num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")
    
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    
    model = create_segformer_model(num_classes=num_classes, model_size=args.model_size)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    optimizer = create_optimizer(model, args.lr)
    print(f"Optimizer: AdamW")
    print(f"Learning rate: {args.lr}")
    
    total_steps = len(train_loader) * args.epochs
    scheduler = create_scheduler(optimizer, total_steps)
    print(f"Scheduler: CosineAnnealingLR")
    
    scaler = GradScaler() if device.type == 'cuda' else None
    
    start_epoch = 1
    best_iou = 0.0
    
    if args.resume:
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = os.path.join(args.save_dir, 'segformer_last_model.pth')
        
        if os.path.exists(checkpoint_path):
            start_epoch, best_iou = load_checkpoint(checkpoint_path, model, optimizer)
        else:
            print(f"  No checkpoint found. Starting from scratch.")
    
    if start_epoch > args.epochs:
        print(f"\n  Model was previously trained for {start_epoch - 1} epochs.")
        print(f"   Current --epochs is set to {args.epochs}.")
        print(f"   Will train for {args.epochs} more epochs (total: {start_epoch - 1 + args.epochs} epochs).")
        end_epoch = start_epoch + args.epochs
    else:
        end_epoch = args.epochs + 1
    
    print("\n" + "="*60)
    print("Starting training...")
    print(f"Training from epoch {start_epoch} to {end_epoch - 1}")
    print("="*60)
    
    for epoch in range(start_epoch, end_epoch):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{end_epoch - 1}")
        print(f"{'='*60}")
        
        train_loss, train_acc, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, end_epoch - 1, scaler
        )
        print(f"\nTraining   - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, IoU: {train_iou:.4f}")
        
        val_loss, val_acc, val_iou = validate(model, val_loader, criterion, device, num_classes)
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, IoU: {val_iou:.4f}")
        
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, best_iou, args.save_dir, f'segformer_checkpoint_epoch_{epoch}.pth')
        
        save_checkpoint(model, optimizer, epoch, best_iou, args.save_dir, 'segformer_last_model.pth')
        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(model, optimizer, epoch, best_iou, args.save_dir, 'segformer_best_model.pth')
            print(f" New best model! IoU: {best_iou:.4f}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation IoU: {best_iou:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()