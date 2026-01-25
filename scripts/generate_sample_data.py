"""Sample data generation for testing and demonstration."""

import json
import os
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_sample_images(output_dir: str, num_images: int = 10) -> List[str]:
    """Create sample images for testing.
    
    Args:
        output_dir: Directory to save images.
        num_images: Number of images to create.
        
    Returns:
        List of created image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = []
    
    # Sample image descriptions
    descriptions = [
        ("A red apple on a white table", (255, 0, 0)),
        ("A blue car parked on the street", (0, 0, 255)),
        ("A green tree in a park", (0, 255, 0)),
        ("A yellow sun in the sky", (255, 255, 0)),
        ("A black cat sitting on a chair", (0, 0, 0)),
        ("A white dog playing in the garden", (255, 255, 255)),
        ("A brown horse in a field", (139, 69, 19)),
        ("A purple flower in a vase", (128, 0, 128)),
        ("An orange sunset over mountains", (255, 165, 0)),
        ("A pink rose in bloom", (255, 192, 203)),
    ]
    
    for i in range(min(num_images, len(descriptions))):
        desc, color = descriptions[i]
        
        # Create a simple colored rectangle with text
        img = Image.new('RGB', (224, 224), color)
        draw = ImageDraw.Draw(img)
        
        # Add text
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw text in the center
        bbox = draw.textbbox((0, 0), desc, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (224 - text_width) // 2
        y = (224 - text_height) // 2
        
        # Use contrasting color for text
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        draw.text((x, y), desc, fill=text_color, font=font)
        
        # Save image
        image_path = os.path.join(output_dir, f"sample_{i:03d}.jpg")
        img.save(image_path)
        image_paths.append(image_path)
    
    return image_paths


def create_vqa_dataset(output_dir: str, image_paths: List[str]) -> str:
    """Create a sample VQA dataset.
    
    Args:
        output_dir: Directory to save dataset.
        image_paths: List of image paths.
        
    Returns:
        Path to the created dataset file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample questions and answers
    vqa_data = {
        "train": [],
        "val": [],
    }
    
    questions_answers = [
        ("What color is the main object?", "red"),
        ("What is in the image?", "apple"),
        ("Where is the object located?", "table"),
        ("What is the object doing?", "sitting"),
        ("How many objects are there?", "one"),
    ]
    
    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        
        for question, answer in questions_answers:
            item = {
                "image": image_name,
                "question": question,
                "answer": answer,
                "image_id": i,
                "question_id": f"{i}_{hash(question) % 10000}",
            }
            
            # Split between train and val
            if i % 4 == 0:
                vqa_data["val"].append(item)
            else:
                vqa_data["train"].append(item)
    
    # Save dataset
    dataset_path = os.path.join(output_dir, "vqa_dataset.json")
    with open(dataset_path, 'w') as f:
        json.dump(vqa_data, f, indent=2)
    
    return dataset_path


def create_caption_dataset(output_dir: str, image_paths: List[str]) -> str:
    """Create a sample image captioning dataset.
    
    Args:
        output_dir: Directory to save dataset.
        image_paths: List of image paths.
        
    Returns:
        Path to the created dataset file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample captions
    captions_data = {
        "train": [],
        "val": [],
    }
    
    sample_captions = [
        "A beautiful colored object in the center of the image",
        "This image shows a simple geometric shape with text",
        "A colorful rectangle with descriptive text",
        "An abstract representation of the described object",
        "A minimalist design with bold colors and text",
    ]
    
    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        
        # Use different captions for variety
        caption = sample_captions[i % len(sample_captions)]
        
        item = {
            "image": image_name,
            "caption": caption,
            "image_id": i,
            "caption_id": f"{i}_{hash(caption) % 10000}",
        }
        
        # Split between train and val
        if i % 4 == 0:
            captions_data["val"].append(item)
        else:
            captions_data["train"].append(item)
    
    # Save dataset
    dataset_path = os.path.join(output_dir, "captions_dataset.json")
    with open(dataset_path, 'w') as f:
        json.dump(captions_data, f, indent=2)
    
    return dataset_path


def create_classification_dataset(output_dir: str, image_paths: List[str]) -> List[str]:
    """Create a sample classification dataset.
    
    Args:
        output_dir: Directory to save dataset.
        image_paths: List of image paths.
        
    Returns:
        List of class labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample class labels
    class_labels = [
        "a photo of a red object",
        "a photo of a blue object", 
        "a photo of a green object",
        "a photo of a yellow object",
        "a photo of a black object",
        "a photo of a white object",
        "a photo of a brown object",
        "a photo of a purple object",
        "a photo of an orange object",
        "a photo of a pink object",
    ]
    
    # Save class labels
    labels_path = os.path.join(output_dir, "class_labels.txt")
    with open(labels_path, 'w') as f:
        for label in class_labels:
            f.write(f"{label}\n")
    
    return class_labels


def main():
    """Generate sample data for testing."""
    print("Generating sample data...")
    
    # Create directories
    data_dir = "data/sample"
    images_dir = os.path.join(data_dir, "images")
    
    # Create sample images
    print("Creating sample images...")
    image_paths = create_sample_images(images_dir, num_images=10)
    print(f"Created {len(image_paths)} sample images")
    
    # Create VQA dataset
    print("Creating VQA dataset...")
    vqa_path = create_vqa_dataset(data_dir, image_paths)
    print(f"Created VQA dataset: {vqa_path}")
    
    # Create captioning dataset
    print("Creating captioning dataset...")
    captions_path = create_caption_dataset(data_dir, image_paths)
    print(f"Created captioning dataset: {captions_path}")
    
    # Create classification labels
    print("Creating classification labels...")
    class_labels = create_classification_dataset(data_dir, image_paths)
    print(f"Created {len(class_labels)} class labels")
    
    print("Sample data generation complete!")
    print(f"Data directory: {data_dir}")
    print(f"Images directory: {images_dir}")


if __name__ == "__main__":
    main()
