from flask import Flask, render_template, request, jsonify, send_file
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os
import io
import base64
import json
from segtransformer.segformer import create_segformer_model
from segtransformer.dataloader import IDDAWDataset

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

model = None
device = None
num_classes = None
color_map = None
label_to_canonical_id = None

CONFIG = {
    'checkpoint_path': './segtransformer/checkpoints/segformer_checkpoint_epoch_100.pth',
    'data_root': './IDDAW',
    'img_width': 512,
    'img_height': 256,
    'model_size': 'b2'
}

def create_color_map(num_classes):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background as black
    return colors

def colorize_mask(mask, color_map):
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label_id in range(len(color_map)):
        colored_mask[mask == label_id] = color_map[label_id]
    
    return colored_mask


def calculate_metrics(pred, target, num_classes, ignore_index=255):
    pred = pred.flatten()
    target = target.flatten()
    
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    
    # Pixel accuracy
    correct = (pred == target).sum()
    total = len(target)
    pixel_acc = correct / (total + 1e-10)
    
    # Per-class IoU
    ious = []
    class_ious = {}
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
            class_ious[cls] = float(iou)
    
    mean_iou = np.mean(ious) if ious else 0.0
    return float(pixel_acc), float(mean_iou), class_ious


def image_to_base64(image_array):
    if len(image_array.shape) == 2:  # Grayscale
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    
    image = Image.fromarray(image_array.astype(np.uint8))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def load_model_global():
    global model, device, num_classes, color_map, label_to_canonical_id
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_classes, label_to_canonical_id, _ = IDDAWDataset.get_class_info(CONFIG['data_root'])
    color_map = create_color_map(num_classes)
    print(f"Loaded {num_classes} classes")
    
    model = create_segformer_model(num_classes=num_classes, model_size=CONFIG['model_size'])
    if os.path.exists(CONFIG['checkpoint_path']):
        checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint['epoch']} (IoU: {checkpoint['best_iou']:.4f})")
    else:
        print(f"Warning: Checkpoint not found at {CONFIG['checkpoint_path']}")
    model = model.to(device)
    model.eval()

@app.route('/')
def index():
    return render_template('index.html', 
                         num_classes=num_classes,
                         device=str(device),
                         config=CONFIG)

@app.route('/get_classes', methods=['GET'])
def get_classes():
    classes = [{'id': v, 'name': k} for k, v in sorted(label_to_canonical_id.items(), key=lambda x: x[1])]
    return jsonify({'classes': classes})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        image = Image.open(file.stream).convert('RGB')
        image_np = np.array(image)
        target_size = (CONFIG['img_width'], CONFIG['img_height'])
        image_resized = cv2.resize(image_np, target_size, interpolation=cv2.INTER_LINEAR)
        
        image_tensor = torch.from_numpy(image_resized).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
        
        pred_colored = colorize_mask(pred, color_map)
        overlay = cv2.addWeighted(image_resized, 0.6, pred_colored, 0.4, 0)
        
        unique, counts = np.unique(pred, return_counts=True)
        total_pixels = pred.size
        distribution = {}
        
        for cls_id, count in zip(unique, counts):
            class_name = [k for k, v in label_to_canonical_id.items() if v == cls_id]
            class_name = class_name[0] if class_name else f"Class {cls_id}"
            distribution[class_name] = {
                'count': int(count),
                'percentage': float((count / total_pixels) * 100)
            }
        
        result = {
            'original': image_to_base64(image_resized),
            'prediction': image_to_base64(pred_colored),
            'overlay': image_to_base64(overlay),
            'distribution': distribution
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_with_gt', methods=['POST'])
def predict_with_gt():
    if 'image' not in request.files or 'ground_truth' not in request.files:
        return jsonify({'error': 'Both image and ground truth required'}), 400
    
    try:
        image = Image.open(request.files['image'].stream).convert('RGB')
        image_np = np.array(image)
        
        gt_image = Image.open(request.files['ground_truth'].stream).convert('L')
        gt_mask = np.array(gt_image)
        
        target_size = (CONFIG['img_width'], CONFIG['img_height'])
        image_resized = cv2.resize(image_np, target_size, interpolation=cv2.INTER_LINEAR)
        gt_mask_resized = cv2.resize(gt_mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        image_tensor = torch.from_numpy(image_resized).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
        
        pred_colored = colorize_mask(pred, color_map)
        gt_colored = colorize_mask(gt_mask_resized, color_map)
        pixel_acc, mean_iou, class_ious = calculate_metrics(pred, gt_mask_resized, num_classes)
        
        class_iou_details = {}
        for cls_id, iou in class_ious.items():
            class_name = [k for k, v in label_to_canonical_id.items() if v == cls_id]
            class_name = class_name[0] if class_name else f"Class {cls_id}"
            class_iou_details[class_name] = iou
        
        result = {
            'original': image_to_base64(image_resized),
            'prediction': image_to_base64(pred_colored),
            'ground_truth': image_to_base64(gt_colored),
            'metrics': {
                'pixel_accuracy': pixel_acc,
                'mean_iou': mean_iou,
                'class_ious': class_iou_details
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json
    
    if 'img_width' in data:
        CONFIG['img_width'] = int(data['img_width'])
    if 'img_height' in data:
        CONFIG['img_height'] = int(data['img_height'])
    
    return jsonify({'status': 'success', 'config': CONFIG})


if __name__ == '__main__':
    print("Loading model...")
    load_model_global()
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)