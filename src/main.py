"""Modernized Multi-modal Vision-Language Model implementation."""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from src.models.vl_models import CLIPWrapper, BLIPWrapper
from src.utils.device import get_device, set_seed
from src.utils.config import Config, load_config
from src.eval.metrics import VLMMetrics


class MultiModalVLM:
    """Main class for Multi-modal Vision-Language Models."""
    
    def __init__(self, config: Config):
        """Initialize the VLM system.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.device = get_device()
        set_seed(config.seed)
        
        # Initialize models
        self.clip_model = None
        self.blip_model = None
        self.metrics = VLMMetrics()
        
        # Load models based on configuration
        self._load_models()
    
    def _load_models(self) -> None:
        """Load vision-language models."""
        if self.config.model.name == "clip":
            self.clip_model = CLIPWrapper(
                model_name=self.config.model.pretrained,
                freeze_vision=self.config.model.freeze_vision,
                freeze_text=self.config.model.freeze_text,
            ).to(self.device)
        
        elif self.config.model.name == "blip":
            self.blip_model = BLIPWrapper(
                model_name=self.config.model.pretrained,
            ).to(self.device)
    
    def zero_shot_classification(
        self,
        image_path: str,
        candidate_labels: List[str],
    ) -> Dict[str, float]:
        """Perform zero-shot classification using CLIP.
        
        Args:
            image_path: Path to the input image.
            candidate_labels: List of candidate class labels.
            
        Returns:
            Dictionary containing predictions and confidence scores.
        """
        if self.clip_model is None:
            raise ValueError("CLIP model not loaded. Please set model.name='clip' in config.")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess inputs
        inputs = self.clip_model.processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs["logits_per_image"]
            probs = logits_per_image.softmax(dim=1)
        
        # Get predictions
        predicted_idx = torch.argmax(probs, dim=1).item()
        predicted_label = candidate_labels[predicted_idx]
        confidence = probs[0, predicted_idx].item()
        
        # Create results
        results = {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'all_probabilities': {
                label: prob.item() 
                for label, prob in zip(candidate_labels, probs[0])
            }
        }
        
        return results
    
    def visual_question_answering(
        self,
        image_path: str,
        question: str,
    ) -> Dict[str, str]:
        """Perform visual question answering using BLIP.
        
        Args:
            image_path: Path to the input image.
            question: Question about the image.
            
        Returns:
            Dictionary containing the answer.
        """
        if self.blip_model is None:
            raise ValueError("BLIP model not loaded. Please set model.name='blip' in config.")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess inputs
        inputs = self.blip_model.processor(
            images=image,
            text=question,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate answer
        with torch.no_grad():
            generated_ids = self.blip_model.generate_answer(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs['attention_mask'],
                max_length=50,
            )
        
        # Decode answer
        answer = self.blip_model.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return {
            'question': question,
            'answer': answer,
            'image_path': image_path,
        }
    
    def image_text_similarity(
        self,
        image_path: str,
        text: str,
    ) -> float:
        """Calculate similarity between image and text using CLIP.
        
        Args:
            image_path: Path to the input image.
            text: Text description.
            
        Returns:
            Similarity score between 0 and 1.
        """
        if self.clip_model is None:
            raise ValueError("CLIP model not loaded. Please set model.name='clip' in config.")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess inputs
        inputs = self.clip_model.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            image_embeds = self.clip_model.encode_image(inputs['pixel_values'])
            text_embeds = self.clip_model.encode_text(
                inputs['input_ids'],
                inputs['attention_mask']
            )
        
        # Calculate similarity
        similarity = self.clip_model.compute_similarity(image_embeds, text_embeds)
        
        return similarity[0, 0].item()
    
    def batch_evaluation(
        self,
        image_paths: List[str],
        texts: List[str],
        task: str = "similarity",
    ) -> Dict[str, List]:
        """Evaluate model on a batch of images and texts.
        
        Args:
            image_paths: List of image paths.
            texts: List of texts (questions, captions, or descriptions).
            task: Task type ('similarity', 'classification', 'vqa').
            
        Returns:
            Dictionary containing evaluation results.
        """
        results = []
        
        for image_path, text in zip(image_paths, texts):
            if task == "similarity":
                score = self.image_text_similarity(image_path, text)
                results.append({
                    'image_path': image_path,
                    'text': text,
                    'similarity': score,
                })
            
            elif task == "classification":
                # For classification, text should be a list of candidate labels
                if isinstance(text, str):
                    candidate_labels = text.split(',')
                else:
                    candidate_labels = text
                
                result = self.zero_shot_classification(image_path, candidate_labels)
                results.append({
                    'image_path': image_path,
                    'candidate_labels': candidate_labels,
                    **result,
                })
            
            elif task == "vqa":
                result = self.visual_question_answering(image_path, text)
                results.append(result)
        
        return {
            'results': results,
            'task': task,
            'num_samples': len(results),
        }
    
    def visualize_results(
        self,
        image_path: str,
        results: Dict,
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize model results.
        
        Args:
            image_path: Path to the input image.
            results: Model results to visualize.
            save_path: Optional path to save the visualization.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Load and display image
        image = Image.open(image_path)
        ax.imshow(image)
        ax.axis('off')
        
        # Add results as text
        if 'predicted_label' in results:
            title = f"Predicted: {results['predicted_label']}\nConfidence: {results['confidence']:.2%}"
            ax.set_title(title, fontsize=14, pad=20)
        
        elif 'answer' in results:
            title = f"Q: {results['question']}\nA: {results['answer']}"
            ax.set_title(title, fontsize=12, pad=20)
        
        elif 'similarity' in results:
            title = f"Similarity: {results['similarity']:.3f}"
            ax.set_title(title, fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """Main function for running the VLM system."""
    parser = argparse.ArgumentParser(description="Multi-modal Vision-Language Models")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--text", type=str, help="Input text")
    parser.add_argument("--task", type=str, default="classification", 
                       choices=["classification", "vqa", "similarity"],
                       help="Task to perform")
    parser.add_argument("--labels", type=str, nargs="+", 
                       help="Candidate labels for classification")
    parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize VLM system
    vlm = MultiModalVLM(config)
    
    # Perform task
    if args.image and args.text:
        if args.task == "classification":
            if not args.labels:
                args.labels = ["a photo of a cat", "a photo of a dog", "a picture of a person"]
            
            results = vlm.zero_shot_classification(args.image, args.labels)
            print(f"Predicted label: {results['predicted_label']}")
            print(f"Confidence: {results['confidence']:.2%}")
            
        elif args.task == "vqa":
            results = vlm.visual_question_answering(args.image, args.text)
            print(f"Question: {results['question']}")
            print(f"Answer: {results['answer']}")
            
        elif args.task == "similarity":
            similarity = vlm.image_text_similarity(args.image, args.text)
            print(f"Similarity score: {similarity:.3f}")
        
        # Visualize results
        vlm.visualize_results(args.image, results)
    
    else:
        print("Please provide both --image and --text arguments.")


if __name__ == "__main__":
    main()
