"""Tests for Multi-modal Vision-Language Models."""

import os
import tempfile
import unittest
from typing import Dict, List

import torch
from PIL import Image
import numpy as np

from src.models.vl_models import CLIPWrapper, BLIPWrapper, MultiModalFusion
from src.utils.device import get_device, set_seed
from src.utils.config import Config
from src.eval.metrics import VLMMetrics


class TestCLIPWrapper(unittest.TestCase):
    """Test cases for CLIP wrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = get_device()
        set_seed(42)
        
        # Create a simple test image
        self.test_image = Image.new('RGB', (224, 224), color='red')
        
        # Test text
        self.test_text = ["a photo of a cat", "a photo of a dog"]
    
    def test_clip_initialization(self):
        """Test CLIP model initialization."""
        model = CLIPWrapper(
            model_name="openai/clip-vit-base-patch32",
            freeze_vision=False,
            freeze_text=False,
        )
        
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.processor)
    
    def test_clip_forward_pass(self):
        """Test CLIP forward pass."""
        model = CLIPWrapper("openai/clip-vit-base-patch32")
        
        # Preprocess inputs
        inputs = model.processor(
            text=self.test_text,
            images=self.test_image,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Check output structure
        self.assertIn('logits_per_image', outputs)
        self.assertIn('logits_per_text', outputs)
        self.assertIn('text_embeds', outputs)
        self.assertIn('image_embeds', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['logits_per_image'].shape, (1, len(self.test_text)))
        self.assertEqual(outputs['logits_per_text'].shape, (len(self.test_text), 1))
    
    def test_clip_encoding(self):
        """Test CLIP encoding functions."""
        model = CLIPWrapper("openai/clip-vit-base-patch32")
        
        # Test image encoding
        inputs = model.processor(images=self.test_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            image_embeds = model.encode_image(pixel_values)
        
        self.assertEqual(image_embeds.shape, (1, 512))  # CLIP ViT-B/32 has 512-dim embeddings
        
        # Test text encoding
        inputs = model.processor(text=self.test_text[0], return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            text_embeds = model.encode_text(input_ids, attention_mask)
        
        self.assertEqual(text_embeds.shape, (1, 512))
    
    def test_clip_similarity(self):
        """Test CLIP similarity computation."""
        model = CLIPWrapper("openai/clip-vit-base-patch32")
        
        # Get embeddings
        image_inputs = model.processor(images=self.test_image, return_tensors="pt")
        text_inputs = model.processor(text=self.test_text[0], return_tensors="pt")
        
        pixel_values = image_inputs['pixel_values'].to(self.device)
        input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            image_embeds = model.encode_image(pixel_values)
            text_embeds = model.encode_text(input_ids, attention_mask)
            similarity = model.compute_similarity(image_embeds, text_embeds)
        
        self.assertEqual(similarity.shape, (1, 1))
        self.assertTrue(-1.0 <= similarity.item() <= 1.0)


class TestBLIPWrapper(unittest.TestCase):
    """Test cases for BLIP wrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = get_device()
        set_seed(42)
        
        # Create a simple test image
        self.test_image = Image.new('RGB', (224, 224), color='blue')
        
        # Test question
        self.test_question = "What color is the image?"
    
    def test_blip_initialization(self):
        """Test BLIP model initialization."""
        model = BLIPWrapper("Salesforce/blip-vqa-base")
        
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.processor)
    
    def test_blip_forward_pass(self):
        """Test BLIP forward pass."""
        model = BLIPWrapper("Salesforce/blip-vqa-base")
        
        # Preprocess inputs
        inputs = model.processor(
            images=self.test_image,
            text=self.test_question,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Check output structure
        self.assertIn('logits', outputs)
        
        # Check output shape
        self.assertEqual(outputs['logits'].shape[0], 1)  # Batch size 1
    
    def test_blip_generation(self):
        """Test BLIP answer generation."""
        model = BLIPWrapper("Salesforce/blip-vqa-base")
        
        # Preprocess inputs
        inputs = model.processor(
            images=self.test_image,
            text=self.test_question,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate answer
        with torch.no_grad():
            generated_ids = model.generate_answer(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs['attention_mask'],
                max_length=20,
            )
        
        # Decode answer
        answer = model.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)


class TestMultiModalFusion(unittest.TestCase):
    """Test cases for multi-modal fusion layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = get_device()
        set_seed(42)
    
    def test_fusion_initialization(self):
        """Test fusion layer initialization."""
        fusion = MultiModalFusion(
            vision_dim=512,
            text_dim=512,
            hidden_dim=256,
            num_heads=8,
            dropout=0.1,
        )
        
        self.assertIsNotNone(fusion.vision_proj)
        self.assertIsNotNone(fusion.text_proj)
        self.assertIsNotNone(fusion.cross_attention)
    
    def test_fusion_forward_pass(self):
        """Test fusion layer forward pass."""
        fusion = MultiModalFusion(
            vision_dim=512,
            text_dim=512,
            hidden_dim=256,
            num_heads=8,
            dropout=0.1,
        ).to(self.device)
        
        # Create test inputs
        batch_size, seq_len = 2, 10
        vision_features = torch.randn(batch_size, seq_len, 512).to(self.device)
        text_features = torch.randn(batch_size, seq_len, 512).to(self.device)
        
        # Forward pass
        output = fusion(vision_features, text_features)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, 256))


class TestVLMMetrics(unittest.TestCase):
    """Test cases for VLM metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = VLMMetrics()
    
    def test_accuracy(self):
        """Test accuracy calculation."""
        predictions = ["cat", "dog", "bird"]
        references = ["cat", "dog", "cat"]
        
        accuracy = self.metrics.accuracy(predictions, references)
        
        self.assertEqual(accuracy, 2/3)  # 2 out of 3 correct
    
    def test_bleu_score(self):
        """Test BLEU score calculation."""
        predictions = ["a cat sitting on a mat"]
        references = [["a cat is sitting on a mat"]]
        
        bleu_scores = self.metrics.bleu_score(predictions, references)
        
        self.assertIn('bleu', bleu_scores)
        self.assertIn('bleu_std', bleu_scores)
        self.assertGreaterEqual(bleu_scores['bleu'], 0.0)
        self.assertLessEqual(bleu_scores['bleu'], 1.0)
    
    def test_rouge_score(self):
        """Test ROUGE score calculation."""
        predictions = ["a cat sitting on a mat"]
        references = ["a cat is sitting on a mat"]
        
        rouge_scores = self.metrics.rouge_score(predictions, references)
        
        self.assertIn('rouge1', rouge_scores)
        self.assertIn('rouge2', rouge_scores)
        self.assertIn('rougeL', rouge_scores)
        
        for score in rouge_scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_retrieval_metrics(self):
        """Test retrieval metrics calculation."""
        # Create similarity matrix and labels
        similarities = torch.tensor([
            [0.9, 0.1, 0.2],  # First image most similar to first text
            [0.1, 0.8, 0.3],  # Second image most similar to second text
        ])
        labels = torch.tensor([0, 1])  # Ground truth labels
        
        retrieval_metrics = self.metrics.retrieval_metrics(similarities, labels)
        
        self.assertIn('R@1', retrieval_metrics)
        self.assertIn('R@5', retrieval_metrics)
        self.assertIn('R@10', retrieval_metrics)
        self.assertIn('MRR', retrieval_metrics)
        
        # All images should be ranked first (R@1 = 1.0)
        self.assertEqual(retrieval_metrics['R@1'], 1.0)


class TestConfig(unittest.TestCase):
    """Test cases for configuration management."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = Config()
        
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.model.name, "clip")
    
    def test_model_config(self):
        """Test model configuration."""
        config = Config()
        
        self.assertEqual(config.model.pretrained, "openai/clip-vit-base-patch32")
        self.assertEqual(config.model.image_size, 224)
        self.assertFalse(config.model.freeze_vision)
        self.assertFalse(config.model.freeze_text)
    
    def test_data_config(self):
        """Test data configuration."""
        config = Config()
        
        self.assertEqual(config.data.batch_size, 32)
        self.assertEqual(config.data.num_workers, 4)
        self.assertEqual(config.data.image_size, 224)
    
    def test_training_config(self):
        """Test training configuration."""
        config = Config()
        
        self.assertEqual(config.training.learning_rate, 1e-4)
        self.assertEqual(config.training.num_epochs, 10)
        self.assertTrue(config.training.mixed_precision)
        self.assertFalse(config.training.use_wandb)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
