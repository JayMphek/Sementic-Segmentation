import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random
from dataloader import IDDAWDataset
from segformer import create_segformer_model
from config import Config


def load_model(checkpoint_path, num_classes, device):
    model = create_segformer_model(num_classes=num_classes, model_size='b2')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Best IoU: {checkpoint['best_iou']:.4f}")
    return model

def create_color_map(num_classes):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  
    return colors

def colorize_mask(mask, color_map):
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label_id in range(len(color_map)):
        colored_mask[mask == label_id] = color_map[label_id]
    
    return colored_mask

def visualize_prediction(image, gt_mask, pred_mask, color_map, save_path=None):
    if torch.is_tensor(image):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
    
    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.cpu().numpy()
    
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    
    gt_colored = colorize_mask(gt_mask, color_map)
    pred_colored = colorize_mask(pred_mask, color_map)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(gt_colored)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_colored)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_metrics(pred, target, num_classes, ignore_index=255):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    
    correct = (pred == target).sum()
    total = len(target)
    pixel_acc = correct / (total + 1e-10)
    
    ious = []
    class_stats = {}
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
            class_stats[cls] = {
                'iou': iou,
                'intersection': intersection,
                'union': union,
                'pixels': target_cls.sum()
            }
    
    mean_iou = np.mean(ious) if ious else 0.0
    return pixel_acc, mean_iou, class_stats


def test_model(model, dataloader, device, num_classes, config, save_visualizations=False, vis_dir='visualizations'):
    model.eval()
    
    all_pixel_accs = []
    all_ious = []
    all_class_stats = {cls: [] for cls in range(num_classes)}
    
    if save_visualizations:
        os.makedirs(vis_dir, exist_ok=True)
        color_map = create_color_map(num_classes)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        
        for batch_idx, (images, masks, json_paths) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            # Calculate metrics for each image in batch
            for i in range(images.size(0)):
                pixel_acc, mean_iou, class_stats = calculate_metrics(
                    preds[i], masks[i], num_classes, config.IGNORE_INDEX
                )
                
                all_pixel_accs.append(pixel_acc)
                all_ious.append(mean_iou)
                
                for cls, stats in class_stats.items():
                    all_class_stats[cls].append(stats['iou'])
                
                # Save visualization
                if save_visualizations and batch_idx < 10:  
                    save_path = os.path.join(vis_dir, f'prediction_{batch_idx}_{i}.png')
                    visualize_prediction(
                        images[i], masks[i], preds[i], color_map, save_path
                    )
            
            pbar.set_postfix({
                'pixel_acc': f'{np.mean(all_pixel_accs):.4f}',
                'mIoU': f'{np.mean(all_ious):.4f}'
            })
    
    # Calculate final statistics
    avg_pixel_acc = np.mean(all_pixel_accs)
    avg_miou = np.mean(all_ious)
    
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Average Pixel Accuracy: {avg_pixel_acc:.4f}")
    print(f"Mean IoU: {avg_miou:.4f}")
    print("\nPer-class IoU:")
    
    for cls in range(num_classes):
        if all_class_stats[cls]:
            class_iou = np.mean(all_class_stats[cls])
            print(f"  Class {cls:2d}: {class_iou:.4f}")
    
    print("="*60)
    return avg_pixel_acc, avg_miou


def predict_single_image(model, image_path, device, target_size=(1024, 512), color_map=None):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Resize
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_resized).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).squeeze(0)
    
    # Resize prediction back to original size
    pred = pred.cpu().numpy()
    pred = cv2.resize(pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Colorize if color map provided
    if color_map is not None:
        pred_colored = colorize_mask(pred, color_map)
        return pred, pred_colored
    
    return pred, None


def main():
    config = Config()
    
    checkpoint_path = os.path.join(config.SAVE_DIR, config.BEST_MODEL_NAME)
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    if config.DEVICE == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif config.DEVICE == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    print("\nLoading test dataset...")
    test_dataset = IDDAWDataset(
        root_dir=config.DATA_ROOT,
        split=config.VAL_SPLIT,  
        target_size=config.TARGET_SIZE
    )
    
    if len(test_dataset) > config.MAX_TEST_SAMPLES:
        indices = random.sample(range(len(test_dataset)), config.MAX_TEST_SAMPLES)
        test_dataset = Subset(test_dataset, indices)
    print(f"Test samples: {len(test_dataset)}")
    
    if isinstance(test_dataset, Subset):
        num_classes = test_dataset.dataset.num_classes
    else:
        num_classes = test_dataset.num_classes
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    model = load_model(checkpoint_path, num_classes, device)
    test_model(
        model, test_loader, device, num_classes, config,
        save_visualizations=True,
        vis_dir='test_visualizations'
    )

if __name__ == '__main__':
    main()