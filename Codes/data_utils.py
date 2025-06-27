# src/utils/data_utils.py
import os
import json
import pickle
import requests
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
import cv2

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class Vocabulary:
    """Vocabulary builder for captions"""
    
    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'
        
    def build_vocabulary(self, captions: List[str]) -> None:
        """Build vocabulary from captions"""
        print("🔤 Building vocabulary...")
        
        # Count word frequencies
        for caption in tqdm(captions, desc="Counting words"):
            tokens = word_tokenize(caption.lower())
            for token in tokens:
                self.word_count[token] += 1
        
        # Add special tokens
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.START_TOKEN: 1,
            self.END_TOKEN: 2,
            self.UNK_TOKEN: 3
        }
        
        # Add words that meet minimum frequency
        idx = 4
        for word, count in self.word_count.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        # Create reverse mapping
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
        print(f"✅ Vocabulary built: {len(self.word2idx)} words")
        print(f"📊 Words filtered (freq < {self.min_freq}): {len(self.word_count) - len(self.word2idx) + 4}")
    
    def encode_caption(self, caption: str, max_length: int) -> List[int]:
        """Convert caption to indices"""
        tokens = word_tokenize(caption.lower())
        indices = [self.word2idx[self.START_TOKEN]]
        
        for token in tokens:
            indices.append(self.word2idx.get(token, self.word2idx[self.UNK_TOKEN]))
        
        indices.append(self.word2idx[self.END_TOKEN])
        
        # Pad or truncate
        if len(indices) < max_length:
            indices.extend([self.word2idx[self.PAD_TOKEN]] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]
        
        return indices
    
    def decode_caption(self, indices: List[int]) -> str:
        """Convert indices back to caption"""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            if word == self.END_TOKEN:
                break
            if word not in [self.PAD_TOKEN, self.START_TOKEN]:
                words.append(word)
        return ' '.join(words)
    
    def save(self, filepath: str):
        """Save vocabulary to file"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_count': dict(self.word_count),
            'min_freq': self.min_freq
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"💾 Vocabulary saved to {filepath}")
    
    def load(self, filepath: str):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.word_count = Counter(vocab_data['word_count'])
        self.min_freq = vocab_data['min_freq']
        print(f"📂 Vocabulary loaded from {filepath}")

class COCODataset(Dataset):
    """COCO dataset for image captioning and segmentation"""
    
    def __init__(self, 
                 coco_dir: str, 
                 split: str = 'train', 
                 task: str = 'captioning',
                 vocabulary: Optional[Vocabulary] = None,
                 max_caption_length: int = 20,
                 image_size: int = 224,
                 transforms_aug: Optional[transforms.Compose] = None):
        
        self.coco_dir = coco_dir
        self.split = split
        self.task = task
        self.vocabulary = vocabulary
        self.max_caption_length = max_caption_length
        self.image_size = image_size
        
        # Setup paths
        self.image_dir = os.path.join(coco_dir, f"{split}2017")
        self.ann_file = os.path.join(coco_dir, "annotations", f"instances_{split}2017.json")
        self.caption_file = os.path.join(coco_dir, "annotations", f"captions_{split}2017.json")
        
        # Load COCO data
        self.coco_caps = COCO(self.caption_file) if os.path.exists(self.caption_file) else None
        self.coco_inst = COCO(self.ann_file) if os.path.exists(self.ann_file) else None
        
        # Get image IDs
        if self.coco_caps:
            self.image_ids = list(self.coco_caps.imgs.keys())
        elif self.coco_inst:
            self.image_ids = list(self.coco_inst.imgs.keys())
        else:
            raise ValueError(f"No annotation files found in {coco_dir}")
        
        # Image transforms
        if transforms_aug is None:
            if task == 'captioning':
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:  # segmentation
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()
                ])
        else:
            self.transform = transforms_aug
        
        print(f"📊 Loaded {len(self.image_ids)} images for {task} ({split})")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        img_info = self.coco_caps.imgs[image_id] if self.coco_caps else self.coco_inst.imgs[image_id]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            image = torch.zeros(3, self.image_size, self.image_size)
        
        if self.task == 'captioning':
            return self._get_captioning_item(image_id, image)
        else:
            return self._get_segmentation_item(image_id, image)
    
    def _get_captioning_item(self, image_id, image):
        """Get captioning data item"""
        # Get captions for this image
        ann_ids = self.coco_caps.getAnnIds(imgIds=image_id)
        captions = [ann['caption'] for ann in self.coco_caps.loadAnns(ann_ids)]
        
        # Randomly select one caption
        caption = np.random.choice(captions)
        
        # Encode caption if vocabulary is provided
        if self.vocabulary:
            caption_encoded = torch.tensor(
                self.vocabulary.encode_caption(caption, self.max_caption_length)
            )
            return image, caption_encoded, caption
        else:
            return image, caption
    
    def _get_segmentation_item(self, image_id, image):
        """Get segmentation data item"""
        # Get annotations for this image
        ann_ids = self.coco_inst.getAnnIds(imgIds=image_id)
        anns = self.coco_inst.loadAnns(ann_ids)
        
        # Create segmentation mask
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        for ann in anns:
            # Get category ID
            cat_id = ann['category_id']
            
            # Create mask for this annotation
            ann_mask = self.coco_inst.annToMask(ann)
            ann_mask = cv2.resize(ann_mask, (self.image_size, self.image_size))
            
            # Add to combined mask
            mask[ann_mask > 0] = cat_id
        
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask

def download_coco_data(coco_dir: str, subset: bool = True):
    """Download COCO dataset"""
    print("📥 Downloading COCO dataset...")
    
    # URLs for COCO 2017
    if subset:
        # Download smaller validation set for testing
        urls = {
            "val_images": "http://images.cocodataset.org/zips/val2017.zip",
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        }
    else:
        # Full dataset
        urls = {
            "train_images": "http://images.cocodataset.org/zips/train2017.zip",
            "val_images": "http://images.cocodataset.org/zips/val2017.zip",
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        }
    
    os.makedirs(coco_dir, exist_ok=True)
    
    for name, url in urls.items():
        print(f"Downloading {name}...")
        filename = os.path.join(coco_dir, url.split('/')[-1])
        
        if not os.path.exists(filename):
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm(
                desc=name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
        
        # Extract if needed
        if filename.endswith('.zip'):
            import zipfile
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(coco_dir)
    
    print("✅ COCO dataset downloaded successfully!")

def create_data_loaders(coco_dir: str, 
                       vocabulary: Vocabulary,
                       task: str = 'captioning',
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    
    # Create datasets
    train_dataset = COCODataset(
        coco_dir=coco_dir,
        split='train',
        task=task,
        vocabulary=vocabulary if task == 'captioning' else None,
        max_caption_length=20,
        image_size=224 if task == 'captioning' else 256
    )
    
    val_dataset = COCODataset(
        coco_dir=coco_dir,
        split='val',
        task=task,
        vocabulary=vocabulary if task == 'captioning' else None,
        max_caption_length=20,
        image_size=224 if task == 'captioning' else 256
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test data utilities
    from config import config
    
    print("🧪 Testing data utilities...")
    
    # Test vocabulary
    test_captions = [
        "A cat sitting on a table",
        "A dog running in the park",
        "A bird flying in the sky"
    ]
    
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocabulary(test_captions)
    
    encoded = vocab.encode_caption("A cat sitting", 10)
    decoded = vocab.decode_caption(encoded)
    print(f"Original: 'A cat sitting'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    
    print("✅ Data utilities test completed!")