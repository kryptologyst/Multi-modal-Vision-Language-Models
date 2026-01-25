"""Configuration management using OmegaConf."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf


@dataclass
class ModelConfig:
    """Configuration for vision-language models."""
    name: str = "clip"
    pretrained: str = "openai/clip-vit-base-patch32"
    image_size: int = 224
    max_text_length: int = 77
    freeze_vision: bool = False
    freeze_text: bool = False


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    text_max_length: int = 77
    augmentation: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    save_every: int = 5
    use_wandb: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    metrics: List[str] = None
    batch_size: int = 64
    save_predictions: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "bleu", "cider", "rouge"]


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    seed: int = 42
    device: str = "auto"
    output_dir: str = "./outputs"
    log_level: str = "INFO"


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Config: Loaded configuration.
    """
    if config_path and os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        return OmegaConf.to_object(cfg)
    else:
        return Config()


def save_config(config: Config, path: str) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration to save.
        path: Path to save configuration.
    """
    cfg = OmegaConf.structured(config)
    OmegaConf.save(cfg, path)
