# src/config.py
import os
from dataclasses import dataclass
from typing import Dict, Any
import torch

@dataclass
class Config:
    """Configuration class for the entire project"""
    
    # Project paths
    PROJECT_DIR: str = "/Volumes/ExternalHDD/ICP/"
    DATA_DIR: str = os.path.join(PROJECT_DIR, "data")
    COCO_DIR: str = os.path.join(DATA_DIR, "MSCOCO")
    CHECKPOINTS_DIR: str = os.path.join(PROJECT_DIR, "checkpoints")
    
    # Device configuration (optimized for M2 Mac)
    DEVICE: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Image Captioning Config
    class Captioning:
        # Model architecture
        ENCODER_DIM: int = 2048  # ResNet feature dimension
        DECODER_DIM: int = 512   # LSTM hidden dimension
        ATTENTION_DIM: int = 512 # Attention layer dimension
        EMBED_DIM: int = 512     # Word embedding dimension
        VOCAB_SIZE: int = 10000  # Vocabulary size
        
        # Training parameters
        BATCH_SIZE: int = 32 if Config.DEVICE == "mps" else 64  # Smaller for M2 Mac
        LEARNING_RATE: float = 1e-4
        NUM_EPOCHS: int = 20
        TEACHER_FORCING_RATIO: float = 0.5
        
        # Data parameters
        MAX_CAPTION_LENGTH: int = 20
        MIN_WORD_FREQ: int = 5
        
        # Image preprocessing
        IMAGE_SIZE: int = 224
        NORMALIZE_MEAN: tuple = (0.485, 0.456, 0.406)
        NORMALIZE_STD: tuple = (0.229, 0.224, 0.225)
    
    # Image Segmentation Config
    class Segmentation:
        # Model architecture
        IN_CHANNELS: int = 3
        OUT_CHANNELS: int = 91  # COCO has 80 classes + background
        BASE_FILTERS: int = 64
        
        # Training parameters
        BATCH_SIZE: int = 16 if Config.DEVICE == "mps" else 32  # Smaller for M2 Mac
        LEARNING_RATE: float = 1e-4
        NUM_EPOCHS: int = 50
        
        # Data parameters
        IMAGE_SIZE: int = 256
        MASK_THRESHOLD: float = 0.5
        
        # Augmentation
        AUGMENT_PROB: float = 0.5
    
    # Evaluation Config
    class Evaluation:
        BLEU_WEIGHTS: tuple = (0.25, 0.25, 0.25, 0.25)
        IOU_THRESHOLD: float = 0.5
        CONFIDENCE_THRESHOLD: float = 0.5
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Get device information for debugging"""
        info = {
            "device": cls.DEVICE,
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        return info
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        dirs = [
            cls.DATA_DIR,
            cls.COCO_DIR,
            cls.CHECKPOINTS_DIR,
            os.path.join(cls.CHECKPOINTS_DIR, "captioning_model"),
            os.path.join(cls.CHECKPOINTS_DIR, "segmentation_model"),
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"✅ Created project directories")
        print(f"📍 Project root: {cls.PROJECT_DIR}")
        print(f"🖥️  Device: {cls.DEVICE}")

# Global config instance
config = Config()

if __name__ == "__main__":
    # Test configuration
    print("🔧 Configuration Test")
    print(f"Device info: {Config.get_device_info()}")
    Config.create_directories()