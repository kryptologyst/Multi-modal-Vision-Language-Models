"""Training script for Multi-modal Vision-Language Models."""

import argparse
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.models.vl_models import CLIPWrapper, BLIPWrapper
from src.data.datasets import VQADataset, ImageCaptionDataset, create_dataloader
from src.eval.metrics import VLMMetrics, evaluate_model
from src.utils.device import get_device, set_seed
from src.utils.config import Config, load_config


class VLMTrainer:
    """Trainer for Vision-Language Models."""
    
    def __init__(self, config: Config):
        """Initialize trainer.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.device = get_device()
        set_seed(config.seed)
        
        # Initialize model
        self.model = self._load_model()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_criterion()
        self.metrics = VLMMetrics()
        
        # Setup logging
        if config.training.use_wandb:
            wandb.init(
                project="multimodal-vlm",
                config=config.__dict__,
            )
    
    def _load_model(self) -> nn.Module:
        """Load the model based on configuration."""
        if self.config.model.name == "clip":
            return CLIPWrapper(
                model_name=self.config.model.pretrained,
                freeze_vision=self.config.model.freeze_vision,
                freeze_text=self.config.model.freeze_text,
            ).to(self.device)
        
        elif self.config.model.name == "blip":
            return BLIPWrapper(
                model_name=self.config.model.pretrained,
            ).to(self.device)
        
        else:
            raise ValueError(f"Unknown model: {self.config.model.name}")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.num_epochs,
        )
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss criterion."""
        return nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader.
            epoch: Current epoch number.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_masks = batch['attention_masks'].to(self.device)
            
            # Forward pass
            if self.config.model.name == "clip":
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    pixel_values=images,
                )
                
                # Compute loss (simplified - in practice, you'd have proper labels)
                logits = outputs['logits_per_image']
                # Create dummy labels for demonstration
                labels = torch.arange(logits.size(0)).to(self.device)
                loss = self.criterion(logits, labels)
            
            elif self.config.model.name == "blip":
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=images,
                    attention_mask=attention_masks,
                )
                loss = outputs['loss'] if outputs['loss'] is not None else torch.tensor(0.0)
            
            # Backward pass
            loss = loss / self.config.training.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
            })
        
        avg_loss = total_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            dataloader: Validation data loader.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move to device
                images = batch['images'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_masks = batch['attention_masks'].to(self.device)
                
                # Forward pass
                if self.config.model.name == "clip":
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_masks,
                        pixel_values=images,
                    )
                    logits = outputs['logits_per_image']
                    labels = torch.arange(logits.size(0)).to(self.device)
                    loss = self.criterion(logits, labels)
                
                elif self.config.model.name == "blip":
                    outputs = self.model(
                        input_ids=input_ids,
                        pixel_values=images,
                        attention_mask=attention_masks,
                    )
                    loss = outputs['loss'] if outputs['loss'] is not None else torch.tensor(0.0)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        return {
            'val_loss': avg_loss,
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Train the model.
        
        Args:
            train_dataloader: Training data loader.
            val_dataloader: Optional validation data loader.
        """
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.num_epochs):
            # Training
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validation
            val_metrics = {}
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            metrics = {**train_metrics, **val_metrics}
            print(f"Epoch {epoch}: {metrics}")
            
            if self.config.training.use_wandb:
                wandb.log(metrics, step=epoch)
            
            # Save checkpoint
            if val_metrics.get('val_loss', float('inf')) < best_val_loss:
                best_val_loss = val_metrics.get('val_loss', float('inf'))
                self.save_checkpoint(epoch, is_best=True)
            
            if epoch % self.config.training.save_every == 0:
                self.save_checkpoint(epoch, is_best=False)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch.
            is_best: Whether this is the best checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.output_dir, 'best_model.pth'))
        else:
            torch.save(checkpoint, os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        print(f"Checkpoint saved at epoch {epoch}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Multi-modal Vision-Language Models")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, help="Path to validation data")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--output_dir", type=str, help="Output directory for checkpoints")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if provided
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Initialize trainer
    trainer = VLMTrainer(config)
    
    # Create datasets
    if config.model.name == "blip":
        train_dataset = VQADataset(
            data_path=args.train_data,
            image_dir=args.image_dir,
            processor=trainer.model.processor,
            split="train",
        )
        
        val_dataset = None
        if args.val_data:
            val_dataset = VQADataset(
                data_path=args.val_data,
                image_dir=args.image_dir,
                processor=trainer.model.processor,
                split="val",
            )
    
    else:  # CLIP
        train_dataset = ImageCaptionDataset(
            data_path=args.train_data,
            image_dir=args.image_dir,
            processor=trainer.model.processor,
            split="train",
        )
        
        val_dataset = None
        if args.val_data:
            val_dataset = ImageCaptionDataset(
                data_path=args.val_data,
                image_dir=args.image_dir,
                processor=trainer.model.processor,
                split="val",
            )
    
    # Create data loaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=config.evaluation.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
    
    # Train
    trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
