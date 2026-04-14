import os
import glob
import cv2
import numpy as np
import torch
import json 
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

class IDDAWDataset(Dataset):
    CLASS_INFO = None

    def __init__(self, root_dir, split='train', target_size=(1024, 512), transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_size = target_size 
        
        if IDDAWDataset.CLASS_INFO is None:
            IDDAWDataset.get_class_info(root_dir)
            
        self.num_classes, self.label_to_canonical_id, self.raw_to_label = IDDAWDataset.CLASS_INFO

        search_path = os.path.join(self.root_dir, self.split, '*', 'rgb', '*', '*_rgb.png')
        self.rgb_files = glob.glob(search_path)
        
        if not self.rgb_files:
            raise FileNotFoundError(f"No RGB files found at {search_path}. Check your root_dir and split name.")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        
        normalized_path = rgb_path.replace(os.path.sep, '/') 
        temp_mask_path = normalized_path.replace('/rgb/', '/gt_labels/')
        final_mask_path = temp_mask_path.replace('_rgb.png', '_labellevel3Ids.png').replace('/', os.path.sep)
        temp_json_path = normalized_path.replace('/rgb/', '/gtSeg/') 
        final_json_path = temp_json_path.replace('_rgb.png', '_mask.json').replace('/', os.path.sep)
        
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        raw_mask = cv2.imread(final_mask_path, cv2.IMREAD_GRAYSCALE) 
        if raw_mask is None:
             raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        try:
            with open(final_json_path, 'r') as f:
                json_data = json.load(f)
        except Exception:
            json_data = {"objects": []} 

        W, H = self.target_size
        image_resized = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        raw_mask_resized = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        local_raw_to_canonical_map = {}
        for obj in json_data['objects']:
            raw_id = obj.get('id')
            label = obj.get('label')
            
            if raw_id is not None and label in self.label_to_canonical_id:
                local_raw_to_canonical_map[raw_id] = self.label_to_canonical_id[label]

        max_raw_id = max(self.raw_to_label.keys()) if self.raw_to_label else 0
        canonical_mask_lookup = np.full(max_raw_id + 1, fill_value=255, dtype=np.uint8) 
        
        for raw_id, canonical_id in local_raw_to_canonical_map.items():
             canonical_mask_lookup[raw_id] = canonical_id

        canonical_mask_resized = np.take(canonical_mask_lookup, raw_mask_resized)
        
        if self.transform:
             image_tensor = self.transform(image_resized)
        else:
             image_tensor = T.ToTensor()(image_resized.astype(np.float32) / 255.0)

        mask_tensor = torch.from_numpy(canonical_mask_resized).long()
        return image_tensor, mask_tensor, final_json_path
    
    @staticmethod
    def get_class_info(root_dir):
        if IDDAWDataset.CLASS_INFO is not None:
            return IDDAWDataset.CLASS_INFO

        raw_to_label = {}
        label_to_canonical_id = {}
        unique_labels = set()
        splits = ['train', 'val']
        
        print("Scanning JSON files to build canonical label mapping...")
        all_json_files = []
        for split in splits:
            search_path = os.path.join(root_dir, split, '*', 'gtSeg', '*', '*_mask.json')
            json_files = glob.glob(search_path)
            all_json_files.extend(json_files)
        
        for json_path in tqdm(all_json_files, desc="Processing JSON files", unit="file"):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if 'objects' in data:
                        for obj in data['objects']:
                            obj_id = obj.get('id')
                            obj_label = obj.get('label')
                            
                            if obj_id is not None and obj_label:
                                unique_labels.add(obj_label)
                                raw_to_label[obj_id] = obj_label 

            except Exception:
                continue
        
        sorted_labels = sorted(list(unique_labels))
        for i, label in enumerate(sorted_labels):
             label_to_canonical_id[label] = i

        num_classes = len(label_to_canonical_id)
        IDDAWDataset.CLASS_INFO = (num_classes, label_to_canonical_id, raw_to_label)
        print(f"✅ Created a canonical mapping for {num_classes} distinct classes.")
        
        print("Canonical Label Mapping (Text -> Stable ID):")
        for label, canonical_id in label_to_canonical_id.items():
             print(f"  - ID {canonical_id:2d}: '{label}'")
        
        return num_classes, label_to_canonical_id, raw_to_label