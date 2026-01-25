"""Vision-Language model implementations."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)


class CLIPWrapper(nn.Module):
    """Wrapper for CLIP model with additional functionality."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze_vision: bool = False,
        freeze_text: bool = False,
    ):
        """Initialize CLIP wrapper.
        
        Args:
            model_name: Name of the CLIP model to load.
            freeze_vision: Whether to freeze vision encoder.
            freeze_text: Whether to freeze text encoder.
        """
        super().__init__()
        
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if freeze_vision:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
                
        if freeze_text:
            for param in self.model.text_model.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through CLIP model.
        
        Args:
            input_ids: Text token IDs.
            attention_mask: Text attention mask.
            pixel_values: Image pixel values.
            
        Returns:
            Dictionary containing logits and embeddings.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        
        return {
            "logits_per_image": outputs.logits_per_image,
            "logits_per_text": outputs.logits_per_text,
            "text_embeds": outputs.text_embeds,
            "image_embeds": outputs.image_embeds,
        }
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings.
        
        Args:
            pixel_values: Image pixel values.
            
        Returns:
            Image embeddings.
        """
        return self.model.get_image_features(pixel_values=pixel_values)
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text to embeddings.
        
        Args:
            input_ids: Text token IDs.
            attention_mask: Text attention mask.
            
        Returns:
            Text embeddings.
        """
        return self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    
    def compute_similarity(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between image and text embeddings.
        
        Args:
            image_embeds: Image embeddings.
            text_embeds: Text embeddings.
            
        Returns:
            Similarity scores.
        """
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = torch.matmul(image_embeds, text_embeds.T)
        return similarity


class BLIPWrapper(nn.Module):
    """Wrapper for BLIP model for VQA and image captioning."""
    
    def __init__(self, model_name: str = "Salesforce/blip-vqa-base"):
        """Initialize BLIP wrapper.
        
        Args:
            model_name: Name of the BLIP model to load.
        """
        super().__init__()
        
        self.model = BlipForQuestionAnswering.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for VQA.
        
        Args:
            input_ids: Question token IDs.
            pixel_values: Image pixel values.
            attention_mask: Question attention mask.
            
        Returns:
            Dictionary containing logits and loss.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if hasattr(outputs, 'loss') else None,
        }
    
    def generate_answer(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 50,
    ) -> torch.Tensor:
        """Generate answer for VQA.
        
        Args:
            input_ids: Question token IDs.
            pixel_values: Image pixel values.
            attention_mask: Question attention mask.
            max_length: Maximum generation length.
            
        Returns:
            Generated answer token IDs.
        """
        return self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )


class MultiModalFusion(nn.Module):
    """Multi-modal fusion layer for combining vision and language features."""
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize fusion layer.
        
        Args:
            vision_dim: Dimension of vision features.
            text_dim: Dimension of text features.
            hidden_dim: Hidden dimension for fusion.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through fusion layer.
        
        Args:
            vision_features: Vision features [batch_size, seq_len, vision_dim].
            text_features: Text features [batch_size, seq_len, text_dim].
            
        Returns:
            Fused features [batch_size, seq_len, hidden_dim].
        """
        # Project features to common dimension
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        # Cross-attention: vision attends to text
        attn_output, _ = self.cross_attention(
            query=vision_proj,
            key=text_proj,
            value=text_proj,
        )
        
        # Residual connection and layer norm
        vision_out = self.norm1(vision_proj + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(vision_out)
        output = self.norm2(vision_out + ffn_output)
        
        return output
