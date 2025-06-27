# src/captioning/train.py
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

# Import custom modules
from model import ImageCaptioningModel
from ..utils.data_utils import COCODataset, Vocabulary, create_data_loaders
from ..config import config
from ..utils.evaluation import calculate_bleu_score

class CaptioningTrainer:
    """Training class for image captioning model with checkpoints and progress tracking"""
    
    def __init__(self, 
                 model: ImageCaptioningModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 vocabulary: Vocabulary,
                 device: str,
                 checkpoint_dir: str,
                 log_dir: str = None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_bleu4 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.bleu_scores = []
        
        # Optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training progress tracking
        self.total_batches_processed = 0
        self.total_images_processed = 0
        
        print(f"🎯 CaptioningTrainer initialized")
        print(f"📁 Checkpoint directory: {checkpoint_dir}")
        print(f"🖥️  Device: {device}")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_filename = f"captioning_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(self.checkpoint_dir, log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Training started")
    
    def setup_optimizer(self, learning_rate=1e-4, weight_decay=1e-5):
        """Setup optimizer and scheduler"""
        # Different learning rates for encoder and decoder
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())
        
        self.optimizer = optim.Adam([
            {'params': encoder_params, 'lr': learning_rate * 0.1},  # Lower LR for pre-trained encoder
            {'params': decoder_params, 'lr': learning_rate}
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=3, verbose=True
        )
        
        self.logger.info(f"✅ Optimizer setup complete - LR: {learning_rate}")
    
    def save_checkpoint(self, epoch, is_best=False, extra_info=None):
        """Save model checkpoint with detailed information"""
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_bleu4': self.best_bleu4,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'bleu_scores': self.bleu_scores,
            'vocabulary': self.vocabulary,
            'total_batches_processed': self.total_batches_processed,
            'total_images_processed': self.total_images_processed,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'attention_dim': self.model.decoder.attention_dim,
                'embed_dim': self.model.decoder.embed_dim,
                'decoder_dim': self.model.decoder.decoder_dim,
                'vocab_size': self.model.vocab_size,
                'encoder_dim': self.model.decoder.encoder_dim
            }
        }
        
        if extra_info:
            checkpoint_data.update(extra_info)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint_data, best_path)
            self.logger.info(f"💾 Best model saved at epoch {epoch} with BLEU-4: {self.best_bleu4:.4f}")
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint_data, latest_path)
        
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Also save vocabulary separately for easy access
        vocab_path = os.path.join(self.checkpoint_dir, 'vocabulary.pkl')
        self.vocabulary.save(vocab_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.best_bleu4 = checkpoint.get('best_bleu4', 0.0)
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.bleu_scores = checkpoint.get('bleu_scores', [])
            self.total_batches_processed = checkpoint.get('total_batches_processed', 0)
            self.total_images_processed = checkpoint.get('total_images_processed', 0)
            
            self.logger.info(f"✅ Checkpoint loaded from epoch {self.current_epoch}")
            self.logger.info(f"📊 Best BLEU-4 so far: {self.best_bleu4:.4f}")
            self.logger.info(f"📈 Total batches processed: {self.total_batches_processed}")
            self.logger.info(f"📈 Total images processed: {self.total_images_processed}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading checkpoint: {e}")
            return False
    
    def train_epoch(self, epoch):
        """Train for one epoch with detailed progress tracking"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{config.Captioning.NUM_EPOCHS}')
        
        for batch_idx, (images, captions, caption_lengths, original_captions) in enumerate(pbar):
            batch_start_time = time.time()
            
            # Move to device
            images = images.to(self.device)
            captions = captions.to(self.device)
            caption_lengths = caption_lengths.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                predictions, targets, decode_lengths, alphas, sort_ind = self.model(
                    images, captions, caption_lengths
                )
                
                # Calculate loss
                # The targets are the next words after each word in the caption
                targets = targets[:, 1:]  # Remove <START> token
                predictions = predictions[:, :-1, :]  # Remove last prediction
                
                # Pack padded sequences
                predictions = nn.utils.rnn.pack_padded_sequence(
                    predictions, decode_lengths, batch_first=True
                ).data
                targets = nn.utils.rnn.pack_padded_sequence(
                    targets, decode_lengths, batch_first=True
                ).data
                
                loss = self.criterion(predictions, targets)
                
                # Add doubly stochastic attention regularization
                if alphas is not None:
                    alpha_regularization = ((1. - alphas.sum(dim=1)) ** 2).mean()
                    loss += alpha_regularization
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                
                # Update weights
                self.optimizer.step()
                
                # Update progress tracking
                batch_loss = loss.item()
                total_loss += batch_loss
                self.total_batches_processed += 1
                self.total_images_processed += images.size(0)
                
                # Calculate batch processing time
                batch_time = time.time() - batch_start_time
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{batch_loss:.4f}',
                    'Avg': f'{total_loss/(batch_idx+1):.4f}',
                    'Time': f'{batch_time:.2f}s',
                    'Images': f'{self.total_images_processed}'
                })
                
                # Log detailed progress
                if batch_idx % 50 == 0:
                    self.logger.info(
                        f"📊 Epoch {epoch}, Batch {batch_idx}/{num_batches} - "
                        f"Loss: {batch_loss:.4f}, "
                        f"Avg Loss: {total_loss/(batch_idx+1):.4f}, "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
                        f"Time: {batch_time:.2f}s, "
                        f"Total Images: {self.total_images_processed}"
                    )
                
                # Tensorboard logging
                if self.writer and batch_idx % 100 == 0:
                    global_step = epoch * num_batches + batch_idx
                    self.writer.add_scalar('Loss/Train_Batch', batch_loss, global_step)
                    self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
                
            except Exception as e:
                self.logger.error(f"❌ Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        self.logger.info(f"✅ Epoch {epoch} completed - Average Loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate model and calculate BLEU score"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        # For BLEU calculation
        references = []
        hypotheses = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Validation Epoch {epoch}')
            
            for batch_idx, (images, captions, caption_lengths, original_captions) in enumerate(pbar):
                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lengths = caption_lengths.to(self.device)
                
                try:
                    # Forward pass for loss calculation
                    predictions, targets, decode_lengths, alphas, sort_ind = self.model(
                        images, captions, caption_lengths
                    )
                    
                    # Calculate loss
                    targets = targets[:, 1:]
                    predictions = predictions[:, :-1, :]
                    
                    predictions_packed = nn.utils.rnn.pack_padded_sequence(
                        predictions, decode_lengths, batch_first=True
                    ).data
                    targets_packed = nn.utils.rnn.pack_padded_sequence(
                        targets, decode_lengths, batch_first=True
                    ).data
                    
                    loss = self.criterion(predictions_packed, targets_packed)
                    total_loss += loss.item()
                    
                    # Generate captions for BLEU calculation (on a subset to save time)
                    if batch_idx % 10 == 0:  # Only process every 10th batch for BLEU
                        for i in range(min(5, images.size(0))):  # Max 5 images per batch
                            # Generate caption
                            generated_caption = self.model.generate_caption(
                                images[i], self.vocabulary, max_length=20, beam_size=3
                            )
                            
                            # Get reference caption (unsort first)
                            original_idx = sort_ind[i].item()
                            reference_caption = original_captions[original_idx]
                            
                            references.append([reference_caption.split()])
                            hypotheses.append(generated_caption.split())
                    
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Avg': f'{total_loss/(batch_idx+1):.4f}'
                    })
                    
                except Exception as e:
                    self.logger.error(f"❌ Validation error in batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # Calculate BLEU score
        bleu4_score = 0.0
        if references and hypotheses:
            try:
                bleu4_score = calculate_bleu_score(references, hypotheses)
                self.bleu_scores.append(bleu4_score)
            except Exception as e:
                self.logger.error(f"❌ BLEU calculation error: {e}")
        
        self.logger.info(f"✅ Validation completed - Loss: {avg_loss:.4f}, BLEU-4: {bleu4_score:.4f}")
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step(avg_loss)
        
        return avg_loss, bleu4_score
    
    def train(self, num_epochs=None, resume_from_checkpoint=None):
        """Main training loop with checkpoint support"""
        if num_epochs is None:
            num_epochs = config.Captioning.NUM_EPOCHS
        
        # Setup optimizer if not already done
        if self.optimizer is None:
            self.setup_optimizer(config.Captioning.LEARNING_RATE)
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            if self.load_checkpoint(resume_from_checkpoint):
                self.logger.info(f"🔄 Resuming training from epoch {self.current_epoch + 1}")
            else:
                self.logger.info("🆕 Starting fresh training")
        
        start_epoch = self.current_epoch + 1
        self.logger.info(f"🚀 Starting training from epoch {start_epoch} to {num_epochs}")
        
        try:
            for epoch in range(start_epoch, num_epochs + 1):
                epoch_start_time = time.time()
                
                # Training phase
                train_loss = self.train_epoch(epoch)
                
                # Validation phase
                val_loss, bleu4_score = self.validate_epoch(epoch)
                
                # Check if this is the best model
                is_best = bleu4_score > self.best_bleu4
                if is_best:
                    self.best_bleu4 = bleu4_score
                
                # Save checkpoint
                self.save_checkpoint(
                    epoch, 
                    is_best=is_best,
                    extra_info={
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'bleu4_score': bleu4_score
                    }
                )
                
                # Update current epoch
                self.current_epoch = epoch
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Comprehensive logging
                self.logger.info(
                    f"🎯 EPOCH {epoch} SUMMARY:\n"
                    f"   📈 Train Loss: {train_loss:.4f}\n"
                    f"   📊 Val Loss: {val_loss:.4f}\n"
                    f"   🏆 BLEU-4: {bleu4_score:.4f}\n"
                    f"   ⭐ Best BLEU-4: {self.best_bleu4:.4f}\n"
                    f"   ⏱️  Epoch Time: {epoch_time/60:.2f} minutes\n"
                    f"   🖼️  Total Images Processed: {self.total_images_processed}\n"
                    f"   📦 Total Batches Processed: {self.total_batches_processed}"
                )
                
                # Tensorboard logging
                if self.writer:
                    self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
                    self.writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
                    self.writer.add_scalar('BLEU/BLEU4', bleu4_score, epoch)
                    self.writer.add_scalar('Time/Epoch_Minutes', epoch_time/60, epoch)
                
                # Plot progress every 5 epochs
                if epoch % 5 == 0:
                    self.plot_progress()
                
        except KeyboardInterrupt:
            self.logger.info("⚠️ Training interrupted by user")
            self.save_checkpoint(
                self.current_epoch, 
                extra_info={'interrupted': True}
            )
        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
            raise
        finally:
            if self.writer:
                self.writer.close()
        
        self.logger.info("🎉 Training completed!")
        return self.best_bleu4
    
    def plot_progress(self):
        """Plot training progress and save to checkpoint directory"""
        if not self.train_losses:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plots
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        if self.val_losses:
            ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # BLEU score plot
        if self.bleu_scores:
            ax2.plot(range(1, len(self.bleu_scores) + 1), self.bleu_scores, 'g-', linewidth=2)
            ax2.set_title('BLEU-4 Score Progress')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('BLEU-4 Score')
            ax2.grid(True)
        
        # Images processed
        ax3.bar(['Total Images'], [self.total_images_processed], color='skyblue')
        ax3.set_title('Total Images Processed')
        ax3.set_ylabel('Number of Images')
        
        # Batches processed
        ax4.bar(['Total Batches'], [self.total_batches_processed], color='lightcoral')
        ax4.set_title('Total Batches Processed')
        ax4.set_ylabel('Number of Batches')
        
        plt.tight_layout()
        plot_path = os.path.join(self.checkpoint_dir, f'training_progress_epoch_{self.current_epoch}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Progress plot saved: {plot_path}")

def main():
    """Main function to start training"""
    print("🚀 Starting Image Captioning Training Pipeline")
    
    # Setup device
    device = config.DEVICE
    print(f"🖥️  Using device: {device}")
    
    # Create directories
    config.create_directories()
    
    # Load or create vocabulary
    vocab_path = os.path.join(config.CHECKPOINTS_DIR, "captioning_model", "vocabulary.pkl")
    
    if os.path.exists(vocab_path):
        print("📂 Loading existing vocabulary...")
        vocabulary = Vocabulary()
        vocabulary.load(vocab_path)
    else:
        print("🔤 Building vocabulary from scratch...")
        # This would require loading COCO captions first
        # For now, we'll create a placeholder
        vocabulary = Vocabulary(min_freq=config.Captioning.MIN_WORD_FREQ)
        # You'll need to build vocabulary from your COCO dataset
        # vocabulary.build_vocabulary(all_captions)
        # vocabulary.save(vocab_path)
    
    # Create data loaders
    print("📊 Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        coco_dir=config.COCO_DIR,
        vocabulary=vocabulary,
        task='captioning',
        batch_size=config.Captioning.BATCH_SIZE,
        num_workers=4
    )
    
    # Create model
    print("🧠 Creating model...")
    model = ImageCaptioningModel(
        attention_dim=config.Captioning.ATTENTION_DIM,
        embed_dim=config.Captioning.EMBED_DIM,
        decoder_dim=config.Captioning.DECODER_DIM,
        vocab_size=len(vocabulary.word2idx),
        encoder_dim=config.Captioning.ENCODER_DIM,
        dropout=0.5,
        device=device
    ).to(device)
    
    # Create trainer
    checkpoint_dir = os.path.join(config.CHECKPOINTS_DIR, "captioning_model")
    log_dir = os.path.join(checkpoint_dir, "logs")
    
    trainer = CaptioningTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocabulary=vocabulary,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Start training
    best_bleu4 = trainer.train(
        num_epochs=config.Captioning.NUM_EPOCHS,
        resume_from_checkpoint=None  # Set to path if resuming
    )
    
    print(f"🎉 Training completed! Best BLEU-4: {best_bleu4:.4f}")

if __name__ == "__main__":
    main()