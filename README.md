# Multi-modal Vision-Language Models

A production-ready implementation of Multi-modal Vision-Language Models (VLMs) supporting CLIP, BLIP, and custom fusion architectures for various vision-language tasks including zero-shot classification, visual question answering, image captioning, and image-text similarity.

## Features

- **Multiple Model Support**: CLIP, BLIP, and custom multi-modal fusion layers
- **Comprehensive Tasks**: Zero-shot classification, VQA, image captioning, image-text similarity
- **Production Ready**: Type hints, comprehensive testing, configuration management
- **Device Agnostic**: Automatic device detection (CUDA → MPS → CPU)
- **Interactive Demo**: Streamlit web application for easy experimentation
- **Evaluation Metrics**: BLEU, ROUGE, CIDEr, accuracy, retrieval metrics
- **Modern Stack**: PyTorch 2.x, Transformers, OmegaConf, Streamlit

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Multi-modal-Vision-Language-Models.git
cd Multi-modal-Vision-Language-Models

# Install dependencies
pip install -r requirements.txt

# Generate sample data for testing
python scripts/generate_sample_data.py
```

### Basic Usage

```python
from src.main import MultiModalVLM
from src.utils.config import Config

# Load configuration
config = Config()

# Initialize VLM system
vlm = MultiModalVLM(config)

# Zero-shot classification
results = vlm.zero_shot_classification(
    image_path="path/to/image.jpg",
    candidate_labels=["a photo of a cat", "a photo of a dog", "a picture of a person"]
)
print(f"Predicted: {results['predicted_label']} (confidence: {results['confidence']:.2%})")

# Visual Question Answering
answer = vlm.visual_question_answering(
    image_path="path/to/image.jpg",
    question="What is in this image?"
)
print(f"Answer: {answer['answer']}")

# Image-text similarity
similarity = vlm.image_text_similarity(
    image_path="path/to/image.jpg",
    text="A beautiful landscape"
)
print(f"Similarity: {similarity:.3f}")
```

### Command Line Interface

```bash
# Zero-shot classification
python src/main.py --image path/to/image.jpg --text "a photo of a cat" --task classification --labels "a photo of a cat" "a photo of a dog"

# Visual Question Answering
python src/main.py --image path/to/image.jpg --text "What is in this image?" --task vqa

# Image-text similarity
python src/main.py --image path/to/image.jpg --text "A beautiful landscape" --task similarity
```

## Interactive Demo

Launch the Streamlit demo application:

```bash
streamlit run demo/app.py
```

The demo provides:
- **Zero-shot Classification**: Upload an image and specify candidate labels
- **Visual Question Answering**: Ask questions about uploaded images
- **Image-Text Similarity**: Compare images with text descriptions
- **Real-time Results**: Interactive visualizations and metrics

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   │   └── vl_models.py   # CLIP, BLIP, fusion layers
│   ├── data/              # Data loading utilities
│   │   └── datasets.py   # VQA, captioning, classification datasets
│   ├── eval/              # Evaluation metrics
│   │   └── metrics.py     # BLEU, ROUGE, CIDEr, accuracy
│   ├── utils/             # Utility functions
│   │   ├── device.py      # Device management, seeding
│   │   └── config.py      # Configuration management
│   └── main.py           # Main application
├── configs/               # Configuration files
│   ├── default.yaml       # Default CLIP configuration
│   └── blip.yaml         # BLIP configuration
├── scripts/               # Utility scripts
│   ├── train.py          # Training script
│   └── generate_sample_data.py  # Sample data generation
├── tests/                 # Test suite
│   └── test_models.py     # Unit tests
├── demo/                  # Demo application
│   └── app.py            # Streamlit demo
├── data/                  # Data directory
├── assets/                # Output assets
└── requirements.txt       # Dependencies
```

## Configuration

The project uses OmegaConf for flexible configuration management:

### Model Configuration
```yaml
model:
  name: "clip"  # Options: clip, blip
  pretrained: "openai/clip-vit-base-patch32"
  image_size: 224
  max_text_length: 77
  freeze_vision: false
  freeze_text: false
```

### Training Configuration
```yaml
training:
  learning_rate: 1e-4
  weight_decay: 1e-4
  num_epochs: 10
  mixed_precision: true
  use_wandb: false
```

## Training

### Prepare Data

The project supports multiple data formats:

**VQA Dataset Format:**
```json
{
  "train": [
    {
      "image": "image1.jpg",
      "question": "What color is the object?",
      "answer": "red",
      "image_id": 0,
      "question_id": "0_1234"
    }
  ],
  "val": [...]
}
```

**Image Captioning Format:**
```json
{
  "train": [
    {
      "image": "image1.jpg",
      "caption": "A beautiful landscape",
      "image_id": 0,
      "caption_id": "0_5678"
    }
  ],
  "val": [...]
}
```

### Training Commands

```bash
# Train CLIP model
python scripts/train.py \
    --config configs/default.yaml \
    --train_data data/vqa_dataset.json \
    --val_data data/vqa_val.json \
    --image_dir data/images \
    --output_dir outputs/clip_model

# Train BLIP model
python scripts/train.py \
    --config configs/blip.yaml \
    --train_data data/captions_dataset.json \
    --val_data data/captions_val.json \
    --image_dir data/images \
    --output_dir outputs/blip_model
```

## Evaluation

### Metrics

The project provides comprehensive evaluation metrics:

- **Accuracy**: Exact match accuracy for classification and VQA
- **BLEU**: Bilingual Evaluation Understudy for text generation
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
- **CIDEr**: Consensus-based Image Description Evaluation
- **Retrieval Metrics**: R@1, R@5, R@10, MRR for retrieval tasks

### Evaluation Commands

```bash
# Evaluate on test set
python src/eval/evaluate.py \
    --model_path outputs/best_model.pth \
    --test_data data/test_dataset.json \
    --image_dir data/images \
    --output_dir results/
```

## Model Architectures

### CLIP (Contrastive Language-Image Pre-training)
- **Use Cases**: Zero-shot classification, image-text similarity, retrieval
- **Strengths**: Strong zero-shot performance, efficient inference
- **Architecture**: Dual-encoder with contrastive learning

### BLIP (Bootstrapping Language-Image Pre-training)
- **Use Cases**: Visual Question Answering, image captioning
- **Strengths**: Strong VQA performance, generative capabilities
- **Architecture**: Encoder-decoder with cross-modal attention

### Multi-Modal Fusion
- **Use Cases**: Custom architectures, fine-tuning
- **Strengths**: Flexible design, attention-based fusion
- **Architecture**: Cross-attention with feed-forward networks

## Performance Benchmarks

### Zero-shot Classification (ImageNet-1K)
| Model | Top-1 Accuracy | Top-5 Accuracy |
|-------|---------------|----------------|
| CLIP ViT-B/32 | 63.2% | 85.1% |
| CLIP ViT-B/16 | 68.3% | 88.9% |

### Visual Question Answering (VQA v2.0)
| Model | Accuracy |
|-------|----------|
| BLIP Base | 78.25% |
| BLIP Large | 82.15% |

### Image Captioning (COCO)
| Model | BLEU-4 | CIDEr | ROUGE-L |
|-------|--------|-------|---------|
| BLIP Base | 22.4 | 78.4 | 52.3 |
| BLIP Large | 25.2 | 85.6 | 55.7 |

## Device Support

The project automatically detects and uses the best available device:

1. **CUDA**: NVIDIA GPUs with CUDA support
2. **MPS**: Apple Silicon Macs with Metal Performance Shaders
3. **CPU**: Fallback for all other systems

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_models.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Formatting

```bash
# Format code with black
black src/ tests/ scripts/

# Lint with ruff
ruff check src/ tests/ scripts/

# Fix linting issues
ruff check src/ tests/ scripts/ --fix
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## API Reference

### MultiModalVLM Class

Main interface for vision-language model operations.

#### Methods

- `zero_shot_classification(image_path, candidate_labels)`: Perform zero-shot classification
- `visual_question_answering(image_path, question)`: Answer questions about images
- `image_text_similarity(image_path, text)`: Compute image-text similarity
- `batch_evaluation(image_paths, texts, task)`: Evaluate on batch of data

### Configuration Classes

- `Config`: Main configuration container
- `ModelConfig`: Model-specific settings
- `DataConfig`: Data loading configuration
- `TrainingConfig`: Training hyperparameters
- `EvaluationConfig`: Evaluation settings

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the CLIP model
- Salesforce for the BLIP model
- Hugging Face for the Transformers library
- The PyTorch team for the deep learning framework

## Citation

If you use this project in your research, please cite:

```bibtex
@software{multimodal_vlm,
  title={Multi-modal Vision-Language Models},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Multi-modal-Vision-Language-Models}
}
```
# Multi-modal-Vision-Language-Models
