import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torchvision import transforms as T
import torch.optim as optim
from tqdm import tqdm 
from unet import UNet
from dataloader import IDDAWDataset 
from metrics import SegmentationLoss, compute_miou 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

def get_args():
    parser = argparse.ArgumentParser(description='Train the U-Net on IDD-AW data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=512, help='Target width of images')
    parser.add_argument('--data_root', type=str, default='../IDDAW', help='Root directory of the IDDAW dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples to use from the combined dataset.')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to specific checkpoint to resume from')
    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_checkpoint(checkpoint_path, model, optimizer):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return 1, 0.0
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_miou = checkpoint.get('miou', 0.0)
    
    print(f"✅ Resumed from epoch {checkpoint['epoch']}")
    print(f"   Previous best mIoU: {best_miou:.4f}")
    print(f"   Starting from epoch {start_epoch}")
    
    return start_epoch, best_miou

def train_model(model, train_loader, val_loader, criterion, optimizer, device, args, start_epoch=1, best_miou=0.0):
    model.train()
    
    if start_epoch > args.epochs:
        print(f"\n⚠️  Model was previously trained for {start_epoch - 1} epochs.")
        print(f"   Current --epochs is set to {args.epochs}.")
        print(f"   Will train for {args.epochs} more epochs (total: {start_epoch - 1 + args.epochs} epochs).")
        end_epoch = start_epoch + args.epochs
    else:
        end_epoch = args.epochs + 1
    
    print(f"\n{'='*60}")
    print(f"Training from epoch {start_epoch} to {end_epoch - 1}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, end_epoch):
        running_loss = 0.0
        pbar = tqdm(train_loader, 
                    desc=f"Epoch {epoch}/{end_epoch - 1} [Train]", 
                    unit="batch",
                    leave=False) 

        for i, (images, masks, _) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device) 

            # 1. Zero the gradients
            optimizer.zero_grad() 
            # 2. Forward pass: Get the model's prediction (logits)
            logits = model(images) 
            # 3. Calculate Loss
            loss = criterion(logits, masks) 
            # 4. Backward Pass: Compute gradient of the loss
            loss.backward() 
            # 5. Optimization Step: Update the model weights
            optimizer.step() 
            
            running_loss += loss.item()
            
            pbar.set_postfix({'Loss': f'{running_loss / (i + 1):.4f}'})
        pbar.close()
                
        val_loss, val_miou = validate_model(model, val_loader, criterion, device, model.n_classes)
        
        print(f"\n--- Epoch {epoch} Complete ---")
        print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {val_miou:.4f}")
        
        # -----------------------------------------------------
        # 1. Save Periodic Checkpoint (for recovery)
        # -----------------------------------------------------
        os.makedirs(args.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.save_dir, f'unet_checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'miou': val_miou
        }, checkpoint_path)
        print(f"Periodic checkpoint saved to {checkpoint_path}")

        # -----------------------------------------------------
        # 2. Save Last Model (for resuming)
        # -----------------------------------------------------
        last_model_path = os.path.join(args.save_dir, 'unet_last_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'miou': val_miou
        }, last_model_path)

        # -----------------------------------------------------
        # 3. Save Best Model (for final deployment)
        # -----------------------------------------------------
        if val_miou > best_miou:
            print(f"✨ Validation mIoU improved from {best_miou:.4f} to {val_miou:.4f}. Saving best model...")
            best_miou = val_miou
            best_model_path = os.path.join(args.save_dir, 'unet_best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'miou': val_miou
            }, best_model_path)
            
    print("\n" + "="*60)
    print("Training finished!")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print("="*60)

def validate_model(model, val_loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    miou_accumulator = 0.0
    
    vbar = tqdm(val_loader, desc="Validating", unit="batch", leave=False)

    with torch.no_grad():
        for images, masks, _ in vbar:
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()
            total_batches += 1

            predictions = torch.argmax(logits, dim=1) 
            
            predictions_np = predictions.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            miou_batch = compute_miou(predictions_np, masks_np, num_classes)
            miou_accumulator += miou_batch
            
            vbar.set_postfix({'Loss': f'{total_loss / total_batches:.4f}', 'mIoU': f'{miou_accumulator / total_batches:.4f}'})
            
    avg_loss = total_loss / total_batches
    avg_miou = miou_accumulator / total_batches
    
    vbar.close()
    model.train() 
    return avg_loss, avg_miou

if __name__ == '__main__':
    set_seed(42)
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Device and Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 2. Get Class Info (FIX: Correctly unpack 3 values)
    num_classes, label_to_canonical_id, _ = IDDAWDataset.get_class_info(args.data_root)

    # 3. Data Loading and Splitting
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    img_width = args.img_size[0] if isinstance(args.img_size, list) else args.img_size
    img_height = img_width // 2 
    target_size = (img_width, img_height)
    
    train_dataset_raw = IDDAWDataset(args.data_root, split='train', target_size=target_size, transform=transform)
    val_dataset_raw = IDDAWDataset(args.data_root, split='val', target_size=target_size, transform=transform)
    full_dataset = ConcatDataset([train_dataset_raw, val_dataset_raw])
    
    print(f"Total images found in train and val splits: {len(full_dataset)}")

    MAX_SAMPLES = args.max_samples
    if len(full_dataset) > MAX_SAMPLES:
        print(f"Limiting dataset to {MAX_SAMPLES} random samples.")
        indices = torch.randperm(len(full_dataset))[:MAX_SAMPLES].tolist()
        limited_dataset = Subset(full_dataset, indices)
    else:
        limited_dataset = full_dataset
    print(f"Total samples for splitting: {len(limited_dataset)}")

    train_size = int(0.70 * len(limited_dataset))
    val_size = int(0.15 * len(limited_dataset))
    test_size = len(limited_dataset) - train_size - val_size  
    print(f"Splitting into Train ({train_size}), Validation ({val_size}), Test ({test_size})")

    train_set, val_set, test_set = random_split(
        limited_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"DataLoader sizes: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

    # 4. Model, Loss, Optimizer Setup
    model = UNet(n_channels=3, n_classes=num_classes).to(device)
    criterion = SegmentationLoss(ignore_index=255) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. Load checkpoint if resuming
    start_epoch = 1
    best_miou = 0.0
    
    if args.resume:
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = os.path.join(args.save_dir, 'unet_last_model.pth')
        
        if os.path.exists(checkpoint_path):
            start_epoch, best_miou = load_checkpoint(checkpoint_path, model, optimizer)
        else:
            print(f"⚠️  No checkpoint found. Starting from scratch.")
    
    # 6. Training Execution
    train_model(model, train_loader, val_loader, criterion, optimizer, device, args, start_epoch, best_miou)