"""Comprehensive demonstration of Multi-modal Vision-Language Models."""

import os
import sys
import tempfile
from typing import Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import MultiModalVLM
from src.utils.config import Config
from src.utils.device import get_device


def create_demo_image(text: str, color: tuple = (255, 0, 0)) -> Image.Image:
    """Create a simple demo image with text.
    
    Args:
        text: Text to display on the image.
        color: RGB color tuple for the background.
        
    Returns:
        PIL Image object.
    """
    img = Image.new('RGB', (224, 224), color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a system font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # Draw text in the center
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (224 - text_width) // 2
    y = (224 - text_height) // 2
    
    # Use contrasting color for text
    text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
    draw.text((x, y), text, fill=text_color, font=font)
    
    return img


def demonstrate_zero_shot_classification(vlm: MultiModalVLM) -> None:
    """Demonstrate zero-shot classification."""
    print("\n" + "="*60)
    print("ZERO-SHOT CLASSIFICATION DEMONSTRATION")
    print("="*60)
    
    # Create demo images
    demo_images = [
        ("A red apple", (255, 0, 0)),
        ("A blue car", (0, 0, 255)),
        ("A green tree", (0, 255, 0)),
    ]
    
    candidate_labels = [
        "a photo of a red object",
        "a photo of a blue object", 
        "a photo of a green object",
        "a photo of a yellow object",
        "a photo of a black object",
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, (description, color) in enumerate(demo_images):
            # Create image
            img = create_demo_image(description, color)
            image_path = os.path.join(temp_dir, f"demo_{i}.jpg")
            img.save(image_path)
            
            # Perform classification
            results = vlm.zero_shot_classification(image_path, candidate_labels)
            
            print(f"\nImage: {description}")
            print(f"Predicted: {results['predicted_label']}")
            print(f"Confidence: {results['confidence']:.2%}")
            print("All probabilities:")
            for label, prob in results['all_probabilities'].items():
                print(f"  {label}: {prob:.3f}")


def demonstrate_visual_question_answering(vlm: MultiModalVLM) -> None:
    """Demonstrate visual question answering."""
    print("\n" + "="*60)
    print("VISUAL QUESTION ANSWERING DEMONSTRATION")
    print("="*60)
    
    # Create demo image
    img = create_demo_image("A beautiful sunset", (255, 165, 0))
    
    questions = [
        "What color is the main object?",
        "What is in this image?",
        "Describe the scene",
        "What time of day is it?",
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        img.save(tmp_file.name)
        image_path = tmp_file.name
        
        try:
            for question in questions:
                print(f"\nQuestion: {question}")
                try:
                    answer = vlm.visual_question_answering(image_path, question)
                    print(f"Answer: {answer['answer']}")
                except Exception as e:
                    print(f"Error: {e}")
                    print("Note: BLIP model may not be loaded. Using CLIP for similarity instead.")
                    
                    # Fallback to similarity
                    similarity = vlm.image_text_similarity(image_path, question)
                    print(f"Image-text similarity: {similarity:.3f}")
        finally:
            os.unlink(image_path)


def demonstrate_image_text_similarity(vlm: MultiModalVLM) -> None:
    """Demonstrate image-text similarity."""
    print("\n" + "="*60)
    print("IMAGE-TEXT SIMILARITY DEMONSTRATION")
    print("="*60)
    
    # Create demo images
    demo_images = [
        ("A red apple", (255, 0, 0)),
        ("A blue car", (0, 0, 255)),
        ("A green tree", (0, 255, 0)),
    ]
    
    text_descriptions = [
        "a red fruit",
        "a blue vehicle",
        "a green plant",
        "something colorful",
        "an abstract shape",
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, (description, color) in enumerate(demo_images):
            # Create image
            img = create_demo_image(description, color)
            image_path = os.path.join(temp_dir, f"similarity_{i}.jpg")
            img.save(image_path)
            
            print(f"\nImage: {description}")
            print("Similarity scores:")
            
            for text_desc in text_descriptions:
                similarity = vlm.image_text_similarity(image_path, text_desc)
                print(f"  '{text_desc}': {similarity:.3f}")


def demonstrate_batch_evaluation(vlm: MultiModalVLM) -> None:
    """Demonstrate batch evaluation."""
    print("\n" + "="*60)
    print("BATCH EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Create multiple demo images
    demo_images = [
        ("A red apple", (255, 0, 0)),
        ("A blue car", (0, 0, 255)),
        ("A green tree", (0, 255, 0)),
        ("A yellow sun", (255, 255, 0)),
    ]
    
    text_descriptions = [
        "a red fruit",
        "a blue vehicle", 
        "a green plant",
        "a yellow star",
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        image_paths = []
        
        # Create images
        for i, (description, color) in enumerate(demo_images):
            img = create_demo_image(description, color)
            image_path = os.path.join(temp_dir, f"batch_{i}.jpg")
            img.save(image_path)
            image_paths.append(image_path)
        
        # Batch evaluation
        results = vlm.batch_evaluation(
            image_paths=image_paths,
            texts=text_descriptions,
            task="similarity"
        )
        
        print(f"Evaluated {results['num_samples']} image-text pairs")
        print("\nResults:")
        for i, result in enumerate(results['results']):
            print(f"  Image {i+1}: {result['similarity']:.3f}")


def demonstrate_model_info(vlm: MultiModalVLM) -> None:
    """Demonstrate model information."""
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Model type: {vlm.config.model.name}")
    print(f"Model pretrained: {vlm.config.model.pretrained}")
    print(f"Image size: {vlm.config.model.image_size}")
    print(f"Max text length: {vlm.config.model.max_text_length}")
    print(f"Freeze vision: {vlm.config.model.freeze_vision}")
    print(f"Freeze text: {vlm.config.model.freeze_text}")
    
    # Model size information
    if vlm.clip_model is not None:
        from src.utils.device import get_model_size
        model_info = get_model_size(vlm.clip_model)
        print(f"\nCLIP Model Size:")
        print(f"  Total parameters: {model_info['total_params_M']:.1f}M")
        print(f"  Trainable parameters: {model_info['trainable_params_M']:.1f}M")
    
    if vlm.blip_model is not None:
        from src.utils.device import get_model_size
        model_info = get_model_size(vlm.blip_model)
        print(f"\nBLIP Model Size:")
        print(f"  Total parameters: {model_info['total_params_M']:.1f}M")
        print(f"  Trainable parameters: {model_info['trainable_params_M']:.1f}M")


def main():
    """Main demonstration function."""
    print("Multi-modal Vision-Language Models - Comprehensive Demo")
    print("=" * 60)
    
    # Load configuration
    config = Config()
    print(f"Configuration loaded: {config.model.name} model")
    
    # Initialize VLM system
    print("Initializing VLM system...")
    try:
        vlm = MultiModalVLM(config)
        print("VLM system initialized successfully!")
    except Exception as e:
        print(f"Error initializing VLM system: {e}")
        print("This might be due to missing model files or network issues.")
        return
    
    # Run demonstrations
    try:
        demonstrate_model_info(vlm)
        demonstrate_zero_shot_classification(vlm)
        demonstrate_image_text_similarity(vlm)
        demonstrate_batch_evaluation(vlm)
        demonstrate_visual_question_answering(vlm)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run the Streamlit demo: streamlit run demo/app.py")
        print("2. Generate sample data: python scripts/generate_sample_data.py")
        print("3. Run tests: python -m pytest tests/ -v")
        print("4. Check the README.md for more detailed usage instructions")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Please check your installation and dependencies.")


if __name__ == "__main__":
    main()
