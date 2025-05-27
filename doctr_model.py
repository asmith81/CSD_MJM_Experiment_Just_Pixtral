# %% [markdown]
"""
# docTR Model Evaluation Notebook

This notebook evaluates the docTR model's performance on invoice data extraction.
It follows the project's notebook handling rules and functional programming approach.
"""

# %% [markdown]
"""
## Setup and Configuration
### Initial Imports
"""

# %%
# Install dependencies from requirements file
import subprocess
import sys
from pathlib import Path

# Determine root directory and requirements file path
try:
    # When running as a script
    current_file = Path(__file__)
    ROOT_DIR = current_file.parent
except NameError:
    # When running in a notebook
    ROOT_DIR = Path.cwd()

requirements_file = ROOT_DIR / "requirements_doctr.txt"

# Install requirements if file exists
if requirements_file.exists():
    print(f"Installing dependencies from {requirements_file}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)])
else:
    raise FileNotFoundError(f"Requirements file not found at {requirements_file}")

# %%
# Built-in Python modules
import os
import re
import time
import json
import logging
from datetime import datetime

# External dependencies
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# docTR specific imports
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.transforms import Resize
from torchvision import transforms

# GPU support
import torch

# %% [markdown]
"""
### Logging Configuration
"""
# %%
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %% [markdown]
"""
### Root Directory Determination
"""
# %%
# Determine root directory
try:
    # When running as a script
    current_file = Path(__file__)
    ROOT_DIR = current_file.parent
    # Verify both required files exist
    if not (ROOT_DIR / "doctr_model.py").exists() or not (ROOT_DIR / "requirements_doctr.txt").exists():
        raise RuntimeError("Could not find both doctr_model.py and requirements_doctr.txt in the same directory")
except NameError:
    # When running in a notebook, look for the files in current directory
    current_dir = Path.cwd()
    if not (current_dir / "doctr_model.py").exists() or not (current_dir / "requirements_doctr.txt").exists():
        raise RuntimeError("Could not find both doctr_model.py and requirements_doctr.txt in the current directory")
    ROOT_DIR = current_dir

sys.path.append(str(ROOT_DIR))

# Create results directory
results_dir = ROOT_DIR / "results"
results_dir.mkdir(exist_ok=True)
logger.info(f"Results will be saved to: {results_dir}")

# %% [markdown]
"""
## CUDA Availability Check
"""

# %%
def check_cuda_availability() -> bool:
    """
    Check if CUDA is available and log the GPU information.
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        
        logger.info(f"CUDA is available with {gpu_count} GPU(s)")
        logger.info(f"Using GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        return True
    else:
        logger.warning("CUDA is not available. Running on CPU mode.")
        return False

# %% [markdown]
"""
## Verify Installation
"""

# %%
def verify_installation():
    """Verify that all required packages are installed correctly."""
    try:
        import doctr
        import torch
        import PIL
        import numpy
        import pandas
        import matplotlib
        import cv2
        
        logger.info("Successfully imported all required packages")
        logger.info(f"docTR version: {doctr.__version__}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import required package: {e}")
        return False

# Verify installation
verify_installation()

# %% [markdown]
"""
## Model Configuration
### Load Model Configuration
"""

# %%
def load_model_config(config_path: str = None) -> dict:
    """
    Load and validate model configuration from YAML file.
    
    Args:
        config_path (str, optional): Path to config file. If None, uses default config.
    
    Returns:
        dict: Model configuration
    """
    if config_path is None:
        config_path = ROOT_DIR / "config" / "doctr_config.yaml"
    
    # Create default config if it doesn't exist
    if not Path(config_path).exists():
        default_config = {
            "model": {
                "detection": {
                    "name": "db_resnet50",
                    "pretrained": True,
                    "min_size": 1024,
                    "max_size": 2048,
                    "available_architectures": {
                        "db_resnet50": {
                            "description": "DBNet with ResNet-50 backbone",
                            "characteristics": "Good balance of accuracy and speed",
                            "use_case": "General purpose document detection"
                        },
                        "db_mobilenet_v3_large": {
                            "description": "DBNet with MobileNetV3-Large backbone",
                            "characteristics": "Faster, optimized for mobile/edge devices",
                            "use_case": "Mobile applications, edge devices"
                        },
                        "linknet_resnet18": {
                            "description": "LinkNet with ResNet-18 backbone",
                            "characteristics": "Fast, lightweight",
                            "use_case": "Real-time applications with moderate accuracy needs"
                        },
                        "linknet_resnet34": {
                            "description": "LinkNet with ResNet-34 backbone",
                            "characteristics": "Better accuracy than ResNet-18, still relatively fast",
                            "use_case": "Real-time applications with higher accuracy needs"
                        },
                        "fast_tiny": {
                            "description": "FAST with tiny configuration",
                            "characteristics": "Very fast, lowest accuracy",
                            "use_case": "Real-time applications with basic detection needs"
                        },
                        "fast_small": {
                            "description": "FAST with small configuration",
                            "characteristics": "Fast, moderate accuracy",
                            "use_case": "Real-time applications with moderate accuracy needs"
                        },
                        "fast_base": {
                            "description": "FAST with base configuration",
                            "characteristics": "Fast, good accuracy",
                            "use_case": "Real-time applications with good accuracy needs"
                        }
                    }
                },
                "recognition": {
                    "name": "crnn_vgg16_bn",
                    "pretrained": True,
                    "vocab": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                    "available_architectures": {
                        "crnn_vgg16_bn": {
                            "description": "CRNN with VGG16-BN backbone",
                            "characteristics": "Good accuracy, moderate speed",
                            "use_case": "General purpose text recognition"
                        },
                        "crnn_mobilenet_v3_small": {
                            "description": "CRNN with MobileNetV3-Small backbone",
                            "characteristics": "Fast, optimized for mobile/edge devices",
                            "use_case": "Mobile applications, edge devices"
                        },
                        "crnn_mobilenet_v3_large": {
                            "description": "CRNN with MobileNetV3-Large backbone",
                            "characteristics": "Better accuracy than small version, still fast",
                            "use_case": "Mobile applications requiring better accuracy"
                        },
                        "sar_resnet31": {
                            "description": "SAR with ResNet-31 backbone",
                            "characteristics": "High accuracy, slower speed",
                            "use_case": "Applications requiring high accuracy"
                        }
                    }
                }
            },
            "preprocessing": {
                "resize": {
                    "min_size": 1024,
                    "max_size": 2048
                },
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            }
        }
        
        # Create config directory if it doesn't exist
        Path(config_path).parent.mkdir(exist_ok=True)
        
        # Save default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created default config at {config_path}")
        return default_config
    
    # Load existing config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

# Load the configuration
config = load_model_config()

# %% [markdown]
"""
### Device Configuration
"""

# %%
def configure_device() -> torch.device:
    """
    Configure and return the appropriate device for model execution.
    
    Returns:
        torch.device: Device to use for model execution
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Set memory management
        torch.cuda.empty_cache()
        # Optional: Set memory fraction if needed
        # torch.cuda.set_per_process_memory_fraction(0.8)
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device

# Configure the device
device = configure_device()

# %% [markdown]
"""
## Model Architecture Selection
"""

# %%
def get_available_architectures() -> dict:
    """
    Get available detection model architectures with their descriptions from config.
    
    Returns:
        dict: Dictionary of available architectures and their descriptions
    """
    if not config or "model" not in config or "detection" not in config["model"] or "available_architectures" not in config["model"]["detection"]:
        raise ValueError("Configuration missing available architectures")
    
    return config["model"]["detection"]["available_architectures"]

def print_architecture_options() -> None:
    """Print available architectures with their descriptions."""
    architectures = get_available_architectures()
    
    logger.info("\nAvailable Detection Model Architectures:")
    logger.info("=" * 80)
    for arch, info in architectures.items():
        logger.info(f"\nArchitecture: {arch}")
        logger.info(f"Description: {info['description']}")
        logger.info(f"Characteristics: {info['characteristics']}")
        logger.info(f"Use Case: {info['use_case']}")
        logger.info("-" * 80)

def select_architecture() -> str:
    """
    Allow user to select the detection model architecture.
    
    Returns:
        str: Selected architecture name
    """
    architectures = get_available_architectures()
    current_arch = config["model"]["detection"]["name"]
    
    # Print available options in a clear format
    print("\nAvailable Detection Model Architectures:")
    print("=" * 80)
    for i, (arch, info) in enumerate(architectures.items(), 1):
        print(f"\n{i}. {arch}")
        print(f"   Description: {info['description']}")
        print(f"   Characteristics: {info['characteristics']}")
        print(f"   Use Case: {info['use_case']}")
        print("-" * 80)
    
    print(f"\nCurrent selection: {current_arch}")
    
    while True:
        print(f"\nEnter the number (1-{len(architectures)}) of your choice:")
        selection = input("> ").strip()
        
        # Handle empty input (keep current)
        if not selection:
            print(f"\nKeeping current selection: {current_arch}")
            return current_arch
        
        # Handle numeric selection
        if selection.isdigit() and 1 <= int(selection) <= len(architectures):
            selected_architecture = list(architectures.keys())[int(selection) - 1]
            print(f"\nSelected architecture: {selected_architecture}")
            print(f"Description: {architectures[selected_architecture]['description']}")
            return selected_architecture
        
        print(f"\nInvalid selection: {selection}")
        print(f"Please enter a number between 1 and {len(architectures)}.")

def update_config_with_architecture(config: dict, architecture: str) -> dict:
    """
    Update configuration with selected architecture.
    
    Args:
        config (dict): Current configuration
        architecture (str): Selected architecture name
    
    Returns:
        dict: Updated configuration
    """
    # Create a copy of the config to avoid modifying the original
    updated_config = config.copy()
    
    # Update the architecture in the config
    updated_config["model"]["detection"]["name"] = architecture
    
    # Update size parameters based on architecture
    if architecture.startswith("fast"):
        # FAST models work better with smaller input sizes
        updated_config["model"]["detection"]["min_size"] = 512
        updated_config["model"]["detection"]["max_size"] = 1024
    elif architecture.startswith("db_mobilenet"):
        # MobileNet models work well with medium sizes
        updated_config["model"]["detection"]["min_size"] = 768
        updated_config["model"]["detection"]["max_size"] = 1536
    else:
        # Default sizes for other architectures
        updated_config["model"]["detection"]["min_size"] = 1024
        updated_config["model"]["detection"]["max_size"] = 2048
    
    return updated_config

def get_available_recognition_models() -> dict:
    """
    Get available recognition model architectures with their descriptions from config.
    
    Returns:
        dict: Dictionary of available recognition models and their descriptions
    """
    if not config or "model" not in config or "recognition" not in config["model"] or "available_architectures" not in config["model"]["recognition"]:
        raise ValueError("Configuration missing available recognition architectures")
    
    return config["model"]["recognition"]["available_architectures"]

def select_recognition_model() -> str:
    """
    Allow user to select the recognition model architecture.
    
    Returns:
        str: Selected architecture name
    """
    models = get_available_recognition_models()
    current_model = config["model"]["recognition"]["name"]
    
    # Print available options in a clear format
    print("\nAvailable Recognition Model Architectures:")
    print("=" * 80)
    for i, (model, info) in enumerate(models.items(), 1):
        print(f"\n{i}. {model}")
        print(f"   Description: {info['description']}")
        print(f"   Characteristics: {info['characteristics']}")
        print(f"   Use Case: {info['use_case']}")
        print("-" * 80)
    
    print(f"\nCurrent selection: {current_model}")
    
    while True:
        print(f"\nEnter the number (1-{len(models)}) of your choice:")
        selection = input("> ").strip()
        
        # Handle empty input (keep current)
        if not selection:
            print(f"\nKeeping current selection: {current_model}")
            return current_model
        
        # Handle numeric selection
        if selection.isdigit() and 1 <= int(selection) <= len(models):
            selected_model = list(models.keys())[int(selection) - 1]
            print(f"\nSelected model: {selected_model}")
            print(f"Description: {models[selected_model]['description']}")
            return selected_model
        
        print(f"\nInvalid selection: {selection}")
        print(f"Please enter a number between 1 and {len(models)}.")

# %%
# Allow user to select detection model architecture
print("\n=== Detection Model Architecture Selection ===")
print("The detection model is responsible for finding text regions in your documents.")
print("Different architectures offer different trade-offs between speed and accuracy.")
print("Please select the architecture that best fits your needs:")
print("=" * 80)

selected_architecture = select_architecture()
config = update_config_with_architecture(config, selected_architecture)
logger.info("\nConfiguration updated with selected architecture.")
logger.info(f"New detection model: {config['model']['detection']['name']}")
logger.info(f"Input size range: {config['model']['detection']['min_size']} - {config['model']['detection']['max_size']}")

# Allow user to select recognition model architecture
print("\n=== Recognition Model Architecture Selection ===")
print("The recognition model is responsible for reading the text in the detected regions.")
print("Different architectures offer different trade-offs between speed and accuracy.")
print("Please select the architecture that best fits your needs:")
print("=" * 80)

selected_recognition = select_recognition_model()
config["model"]["recognition"]["name"] = selected_recognition
logger.info("\nConfiguration updated with selected recognition model.")
logger.info(f"New recognition model: {config['model']['recognition']['name']}")

# %% [markdown]
"""
## Model Initialization
### Initialize Detection Model
"""

# %%
def print_model_summary(model_info: dict) -> None:
    """
    Print a summary of the model configuration and details.
    
    Args:
        model_info (dict): Dictionary containing model information
    """
    logger.info("\nModel Summary:")
    logger.info("=" * 80)
    for key, value in model_info.items():
        if key == "vocab":
            logger.info(f"Vocabulary size: {len(value)}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 80)

def initialize_detection_model(config: dict, device: torch.device) -> tuple:
    """
    Initialize docTR detection model with specific configuration.
    
    Args:
        config (dict): Model configuration
        device (torch.device): Device to use for model execution
    
    Returns:
        tuple: (detection_model, model_info)
    """
    try:
        # Get detection model configuration
        det_config = config["model"]["detection"]
        reco_config = config["model"]["recognition"]
        model_name = det_config["name"]
        reco_name = reco_config["name"]
        
        # Log model initialization
        logger.info(f"Initializing detection model: {model_name}")
        logger.info(f"Using recognition model: {reco_name}")
        logger.info(f"Using device: {device}")
        
        # Initialize detection model with specific architecture
        detection_model = ocr_predictor(
            det_arch=model_name,
            reco_arch=reco_name,  # Use selected recognition model
            pretrained=det_config["pretrained"],
            assume_straight_pages=True,  # Optimize for straight document pages
            export_as_straight_boxes=True  # Export boxes as straight rectangles
        )
        
        # Move model to specified device
        if device.type == "cuda":
            detection_model.det_predictor.to(device)
            detection_model.reco_predictor.to(device)
        
        # Get model information
        model_info = {
            "name": model_name,
            "recognition_model": reco_name,
            "pretrained": det_config["pretrained"],
            "device": str(device),
            "min_size": det_config["min_size"],
            "max_size": det_config["max_size"]
        }
        
        # Log model details
        logger.info(f"Model architecture: {model_name}")
        logger.info(f"Recognition model: {reco_name}")
        logger.info(f"Pretrained: {det_config['pretrained']}")
        logger.info(f"Input size range: {det_config['min_size']} - {det_config['max_size']}")
        
        # Log memory usage if using CUDA
        if device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # Convert to MB
            memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)    # Convert to MB
            logger.info(f"GPU Memory allocated: {memory_allocated:.2f} MB")
            logger.info(f"GPU Memory reserved: {memory_reserved:.2f} MB")
        
        return detection_model, model_info
        
    except Exception as e:
        logger.error(f"Error initializing detection model: {e}")
        raise

# %%
# Initialize detection model
detection_model, detection_info = initialize_detection_model(config, device)

# %% [markdown]
"""
### Initialize Recognition Model
"""

# %%
def initialize_recognition_model(config: dict, device: torch.device) -> tuple:
    """
    Initialize docTR recognition model with specific configuration.
    
    Args:
        config (dict): Model configuration
        device (torch.device): Device to use for model execution
    
    Returns:
        tuple: (recognition_model, model_info)
    """
    try:
        # Get recognition model configuration
        reco_config = config["model"]["recognition"]
        model_name = reco_config["name"]
        
        # Log model initialization
        logger.info(f"Initializing recognition model: {model_name}")
        logger.info(f"Using device: {device}")
        
        # Initialize recognition model with specific architecture
        from doctr.models import recognition_predictor
        recognition_model = recognition_predictor(
            arch=model_name,
            pretrained=reco_config["pretrained"]
        )
        
        # Move model to device
        if device.type == "cuda":
            recognition_model.to(device)
        
        # Get model information
        model_info = {
            "name": model_name,
            "pretrained": reco_config["pretrained"],
            "device": str(device),
            "vocab": reco_config["vocab"]
        }
        
        # Log model details
        logger.info(f"Model architecture: {model_name}")
        logger.info(f"Pretrained: {reco_config['pretrained']}")
        logger.info(f"Vocabulary size: {len(reco_config['vocab'])}")
        
        # Log memory usage if using CUDA
        if device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # Convert to MB
            memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)    # Convert to MB
            logger.info(f"GPU Memory allocated: {memory_allocated:.2f} MB")
            logger.info(f"GPU Memory reserved: {memory_reserved:.2f} MB")
        
        return recognition_model, model_info
        
    except Exception as e:
        logger.error(f"Error initializing recognition model: {e}")
        raise

# %%
# Initialize recognition model
recognition_model, recognition_info = initialize_recognition_model(config, device)

# %% [markdown]
"""
### Model Warm-up
"""

# %%
def warm_up_model(model: ocr_predictor, config: dict) -> None:
    """
    Warm up the model with a dummy input to ensure proper initialization.
    
    Args:
        model: Initialized OCR predictor model
        config (dict): Model configuration
    """
    try:
        logger.info("Warming up model...")
        
        # Create a dummy input image (white background)
        dummy_image = Image.new('RGB', (config["model"]["detection"]["min_size"], 
                                      config["model"]["detection"]["min_size"]), 
                              color='white')
        
        # Convert to tensor and add batch dimension
        dummy_tensor = transforms.ToTensor()(dummy_image).unsqueeze(0)
        
        # Move to device if using CUDA
        if torch.cuda.is_available():
            dummy_tensor = dummy_tensor.cuda()
        
        # Run inference
        with torch.no_grad():
            _ = model(dummy_tensor)
        
        logger.info("Model warm-up completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model warm-up: {e}")
        raise

# %%
# Warm up both models
warm_up_model(detection_model, config)
warm_up_model(recognition_model, config)

# %% [markdown]
"""
## Image Processing Configuration
### Define Preprocessing Pipeline
"""

# %%
def create_preprocessing_pipeline(config: dict) -> transforms.Compose:
    """
    Create preprocessing pipeline for document images.
    
    Args:
        config (dict): Model configuration
    
    Returns:
        transforms.Compose: Preprocessing pipeline
    """
    try:
        # Get preprocessing configuration
        preprocess_config = config["preprocessing"]
        
        # Create transforms
        transform_list = [
            # Convert PIL Image to tensor
            transforms.ToTensor(),
            
            # Resize transform with preserve_aspect_ratio
            Resize(
                preprocess_config["resize"]["min_size"],
                interpolation=2,  # BILINEAR
                preserve_aspect_ratio=True,
                symmetric_pad=True
            ),
            
            # Normalize transform with proper mean/std values
            transforms.Normalize(
                mean=preprocess_config["normalize"]["mean"],
                std=preprocess_config["normalize"]["std"],
                inplace=True  # Perform normalization in-place for efficiency
            )
        ]
        
        # Create pipeline using torchvision's Compose
        pipeline = transforms.Compose(transform_list)
        
        # Log pipeline configuration
        logger.info("\nPreprocessing Pipeline Configuration:")
        logger.info(f"Resize: {preprocess_config['resize']['min_size']}")
        logger.info(f"Normalize: mean={preprocess_config['normalize']['mean']}, std={preprocess_config['normalize']['std']}")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Error creating preprocessing pipeline: {e}")
        raise

def process_image(image: Image.Image, pipeline: transforms.Compose) -> torch.Tensor:
    """
    Process a single image using the preprocessing pipeline.
    
    Args:
        image (Image.Image): Input image
        pipeline (transforms.Compose): Preprocessing pipeline
    
    Returns:
        torch.Tensor: Processed image tensor
    """
    try:
        # Apply transforms
        processed = pipeline(image)
        
        # Add batch dimension if needed
        if len(processed.shape) == 3:
            processed = processed.unsqueeze(0)
        
        return processed
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

# %%
# Create preprocessing pipeline
preprocessing_pipeline = create_preprocessing_pipeline(config)

# %% [markdown]
"""
### Test Preprocessing Pipeline
"""

# %%
def test_preprocessing_pipeline(pipeline: transforms.Compose) -> None:
    """
    Test the preprocessing pipeline with a sample image.
    
    Args:
        pipeline (transforms.Compose): Preprocessing pipeline
    """
    try:
        # Create a sample image (white background with black text)
        sample_image = Image.new('RGB', (800, 600), color='white')
        
        # Process the image
        processed = process_image(sample_image, pipeline)
        
        # Log results
        logger.info("\nPreprocessing Pipeline Test:")
        logger.info(f"Input image size: {sample_image.size}")
        logger.info(f"Processed tensor shape: {processed.shape}")
        logger.info(f"Processed tensor range: [{processed.min():.3f}, {processed.max():.3f}]")
        
    except Exception as e:
        logger.error(f"Error testing preprocessing pipeline: {e}")
        raise

# Test the pipeline
test_preprocessing_pipeline(preprocessing_pipeline)

# %% [markdown]
"""
### Define Post-processing Pipeline
"""

# %%
def create_postprocessing_pipeline() -> dict:
    """
    Create post-processing pipeline for text extraction and field mapping.
    Uses docTR's detection model output structure to identify key-value pairs.
    
    Returns:
        dict: Dictionary of post-processing functions
    """
    def clean_text(text: str) -> str:
        """
        Clean extracted text by removing unwanted characters and normalizing whitespace.
        
        Args:
            text (str): Raw extracted text
        
        Returns:
            str: Cleaned text
        """
        # Remove special characters but keep basic punctuation
        text = ''.join(c for c in text if c.isprintable() or c.isspace())
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()

    def find_key_value_pairs(detection_result) -> dict:
        """
        Find key-value pairs in the detection result based on spatial relationships.
        
        Args:
            detection_result: docTR Document object containing detection results
        
        Returns:
            dict: Extracted key-value pairs
        """
        # Get all text blocks with their positions
        blocks = []
        for page in detection_result.pages:
            for block in page.blocks:
                # Get all words and their confidences
                words = []
                confidences = []
                for line in block.lines:
                    for word in line.words:
                        words.append(word.value)
                        confidences.append(word.confidence)
                
                # Combine text from all words
                block_text = " ".join(words)
                text = clean_text(block_text)
                
                if text:  # Only include non-empty blocks
                    # Calculate average confidence for the block
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    blocks.append({
                        'text': text,
                        'bbox': block.geometry,  # [x1, y1, x2, y2]
                        'confidence': avg_confidence,
                        'line_id': getattr(block, 'line_id', -1),  # Get line ID if available
                        'block_id': getattr(block, 'block_id', -1)  # Get block ID if available
                    })
        
        # Sort blocks by vertical position (top to bottom)
        blocks.sort(key=lambda x: x['bbox'][1])
        
        # Find key-value pairs
        extracted_data = {}
        i = 0
        while i < len(blocks):
            current_block = blocks[i]
            text = current_block['text'].lower()
            
            # Check if this block might be a key
            if any(keyword in text for keyword in ['work order', 'wo', 'order', 'total', 'amount', 'cost']):
                # Look for value in the next block
                if i + 1 < len(blocks):
                    next_block = blocks[i + 1]
                    # Check if next block is to the right or below
                    if (next_block['bbox'][0] > current_block['bbox'][2] or  # Right
                        next_block['bbox'][1] > current_block['bbox'][1]):   # Below
                        
                        value = next_block['text']
                        confidence = min(current_block['confidence'], next_block['confidence'])
                        
                        # Map to standardized keys
                        if any(keyword in text for keyword in ['work order', 'wo', 'order']):
                            extracted_data['work_order'] = value
                            extracted_data['work_order_confidence'] = confidence
                            extracted_data['work_order_line_id'] = current_block.get('line_id', -1)
                        elif any(keyword in text for keyword in ['total', 'amount', 'cost']):
                            try:
                                # Clean the value string by removing currency symbols and handling commas
                                cleaned_value = value.replace('$', '').replace(',', '').strip()
                                # Convert to float
                                value_float = float(cleaned_value)
                                extracted_data['total_cost'] = value_float
                                extracted_data['total_cost_confidence'] = confidence
                                extracted_data['total_cost_line_id'] = current_block.get('line_id', -1)
                            except ValueError as e:
                                logger.warning(f"Could not convert cost value '{value}': {str(e)}")
                                # Store the original string value if conversion fails
                                extracted_data['total_cost'] = value
                                extracted_data['total_cost_confidence'] = confidence
                                extracted_data['total_cost_line_id'] = current_block.get('line_id', -1)
            
            i += 1
        
        return extracted_data

    def format_output(extracted_data: dict) -> dict:
        """
        Format extracted data into standardized output.
        
        Args:
            extracted_data (dict): Raw extracted data
        
        Returns:
            dict: Formatted output
        """
        return {
            "work_order": extracted_data.get("work_order", ""),
            "work_order_confidence": extracted_data.get("work_order_confidence", 0.0),
            "work_order_line_id": extracted_data.get("work_order_line_id", -1),
            "total_cost": extracted_data.get("total_cost", 0.0),
            "total_cost_confidence": extracted_data.get("total_cost_confidence", 0.0),
            "total_cost_line_id": extracted_data.get("total_cost_line_id", -1),
            "timestamp": datetime.now().isoformat(),
            "raw_blocks": extracted_data.get("raw_blocks", [])
        }

    return {
        "clean_text": clean_text,
        "find_key_value_pairs": find_key_value_pairs,
        "format_output": format_output
    }

# %%
# Create post-processing pipeline
postprocessing_pipeline = create_postprocessing_pipeline()

# %% [markdown]
"""
### Test Post-processing Pipeline
"""

# %%
def test_postprocessing_pipeline(pipeline: dict) -> None:
    """
    Test the post-processing pipeline with a simulated docTR Document object.
    
    Args:
        pipeline (dict): Post-processing pipeline functions
    """
    try:
        # Create a simulated docTR Document object
        from doctr.documents import Document, Page, Block, Line, Word
        
        # Create words
        words = [
            Word("Work", 0.95, [100, 100, 150, 120]),
            Word("Order:", 0.94, [160, 100, 200, 120]),
            Word("12345", 0.98, [210, 100, 280, 120]),
            Word("Total", 0.93, [100, 150, 150, 170]),
            Word("Amount:", 0.94, [160, 150, 220, 170]),
            Word("$1,234.56", 0.97, [230, 150, 300, 170])
        ]
        
        # Create lines
        lines = [
            Line([words[0], words[1]], 0.95, [100, 100, 200, 120]),
            Line([words[2]], 0.98, [210, 100, 280, 120]),
            Line([words[3], words[4]], 0.94, [100, 150, 220, 170]),
            Line([words[5]], 0.97, [230, 150, 300, 170])
        ]
        
        # Create blocks
        blocks = [
            Block([lines[0], lines[1]], 0.96, [100, 100, 280, 120]),
            Block([lines[2], lines[3]], 0.95, [100, 150, 300, 170])
        ]
        
        # Create page
        page = Page(blocks, [0, 0, 800, 600])
        
        # Create document
        doc = Document([page])
        
        # Process detection result
        extracted_data = pipeline["find_key_value_pairs"](doc)
        logger.info("\nPost-processing Pipeline Test:")
        logger.info(f"Extracted data: {json.dumps(extracted_data, indent=2)}")
        
        # Format output
        formatted_output = pipeline["format_output"](extracted_data)
        logger.info(f"Formatted output: {json.dumps(formatted_output, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error testing post-processing pipeline: {e}")
        raise

# Test the pipeline
test_postprocessing_pipeline(postprocessing_pipeline)

# %% [markdown]
"""
## Single Image Test
### Test Function
"""

# %%
def test_single_image(
    image_path: str,
    detection_model: ocr_predictor,
    preprocessing_pipeline: transforms.Compose,
    postprocessing_pipeline: dict,
    max_display_size: tuple = (800, 600)
) -> dict:
    """
    Test the OCR pipeline on a single image.
    
    Args:
        image_path (str): Path to the test image
        detection_model: Initialized detection model
        preprocessing_pipeline: Preprocessing pipeline
        postprocessing_pipeline: Post-processing pipeline
        max_display_size (tuple): Maximum size for displaying the image
    
    Returns:
        dict: Processed results
    """
    try:
        # Load and display original image
        original_image = Image.open(image_path)
        # Convert to RGB if not already
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
            
        logger.info(f"\nProcessing image: {image_path}")
        logger.info(f"Original image size: {original_image.size}")
        logger.info(f"Image mode: {original_image.mode}")
        
        # Create a copy for display
        display_image = original_image.copy()
        # Calculate scaling factors before resizing
        scale_x = max_display_size[0] / original_image.width
        scale_y = max_display_size[1] / original_image.height
        scale_factor = min(scale_x, scale_y)  # Use the smaller scale to maintain aspect ratio
        
        # Resize for display while maintaining aspect ratio
        new_width = int(original_image.width * scale_factor)
        new_height = int(original_image.height * scale_factor)
        display_image = display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Preprocess image for model
        processed_image = preprocessing_pipeline(original_image)
        logger.info(f"Processed image shape before unsqueeze: {processed_image.shape}")
        
        # Ensure the processed image has the correct shape [B, C, H, W]
        if len(processed_image.shape) == 3:
            processed_image = processed_image.unsqueeze(0)  # Add batch dimension if needed
            logger.info(f"Processed image shape after unsqueeze: {processed_image.shape}")
        
        # Move to device if using CUDA
        if torch.cuda.is_available():
            processed_image = processed_image.cuda()
        
        # Run detection and recognition
        logger.info("Running detection and recognition...")
        start_time = time.time()
        with torch.no_grad():
            detection_result = detection_model(processed_image)
        processing_time = time.time() - start_time
        
        # Process results
        extracted_data = postprocessing_pipeline["find_key_value_pairs"](detection_result)
        formatted_output = postprocessing_pipeline["format_output"](extracted_data)
        
        # Log results
        logger.info(f"\nProcessing time: {processing_time:.2f} seconds")
        logger.info("\nExtracted Data:")
        logger.info(json.dumps(formatted_output, indent=2))
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(display_image)
        plt.axis('off')
        plt.title("Original Image (Resized for Display)")
        
        # Add extracted text as annotation
        text_info = []
        for page in detection_result.pages:
            for block in page.blocks:
                # Get block coordinates
                # Geometry is in format [x1, y1, x2, y2]
                if isinstance(block.geometry, (list, tuple)) and len(block.geometry) == 4:
                    x1, y1, x2, y2 = block.geometry
                else:
                    # If geometry is in a different format, try to get bounding box
                    try:
                        # Get all word coordinates
                        word_coords = []
                        for line in block.lines:
                            for word in line.words:
                                if isinstance(word.geometry, (list, tuple)) and len(word.geometry) == 4:
                                    word_coords.append(word.geometry)
                        
                        if word_coords:
                            # Calculate block bounding box from word coordinates
                            x1 = min(coord[0] for coord in word_coords)
                            y1 = min(coord[1] for coord in word_coords)
                            x2 = max(coord[2] for coord in word_coords)
                            y2 = max(coord[3] for coord in word_coords)
                        else:
                            # Skip this block if we can't get coordinates
                            continue
                    except Exception as e:
                        logger.warning(f"Could not process block geometry: {e}")
                        continue
                
                # Scale coordinates to match the resized display image
                x1, x2 = x1 * scale_factor, x2 * scale_factor
                y1, y2 = y1 * scale_factor, y2 * scale_factor
                
                # Draw rectangle
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, color='red', linewidth=2
                )
                plt.gca().add_patch(rect)
                
                # Get text from all words in all lines in the block
                block_text = " ".join(
                    word.value for line in block.lines 
                    for word in line.words
                )
                text_info.append(f"Text: {block_text}\nConf: {block.confidence:.2f}")
        
        # Add text box with extracted information
        text_box = "\n".join(text_info)
        plt.figtext(
            0.02, 0.02, text_box,
            bbox=dict(facecolor='white', alpha=0.8),
            fontsize=8
        )
        
        plt.tight_layout()
        plt.show()
        
        return formatted_output
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

# %%
# Test with a sample image
test_image_path = ROOT_DIR / "data" / "images" / "1017.jpg"
if test_image_path.exists():
    results = test_single_image(
        str(test_image_path),
        detection_model,
        preprocessing_pipeline,
        postprocessing_pipeline
    )
else:
    logger.warning(f"Test image not found at {test_image_path}")
    logger.info("Please ensure the image file exists at the specified path.")

# %% [markdown]
"""
## Batch Processing
### Batch Processing Function
"""

# %%
def process_batch(
    image_dir: str,
    detection_model: ocr_predictor,
    preprocessing_pipeline: transforms.Compose,
    postprocessing_pipeline: dict,
    output_dir: str = None,
    batch_size: int = 10,
    save_interval: int = 5
) -> dict:
    """
    Process a batch of images and save results incrementally.
    
    Args:
        image_dir (str): Directory containing images to process
        detection_model: Initialized detection model
        preprocessing_pipeline: Preprocessing pipeline
        postprocessing_pipeline: Post-processing pipeline
        output_dir (str): Directory to save results (defaults to results/batch_processing)
        batch_size (int): Number of images to process before saving
        save_interval (int): Number of images to process before showing progress
    
    Returns:
        dict: Summary of processing results
    """
    try:
        # Set up output directory
        if output_dir is None:
            output_dir = ROOT_DIR / "results" / "batch_processing"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
            image_files.extend(list(Path(image_dir).glob(f'*{ext.upper()}')))
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        logger.info(f"\nFound {len(image_files)} images to process")
        
        # Initialize results tracking
        results = {
            'processed': [],
            'failed': [],
            'start_time': datetime.now().isoformat(),
            'total_images': len(image_files),
            'successful': 0,
            'failed_count': 0
        }
        
        # Process images in batches
        current_batch = []
        start_time = time.time()
        
        for i, image_path in enumerate(tqdm.tqdm(image_files, desc="Processing images")):
            try:
                # Process single image
                image_result = test_single_image(
                    str(image_path),
                    detection_model,
                    preprocessing_pipeline,
                    postprocessing_pipeline
                )
                
                # Add to current batch
                current_batch.append({
                    'image_path': str(image_path),
                    'result': image_result,
                    'processing_time': time.time() - start_time
                })
                
                results['successful'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results['failed'].append({
                    'image_path': str(image_path),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                results['failed_count'] += 1
                continue
            
            # Save batch if batch_size reached
            if len(current_batch) >= batch_size:
                batch_file = output_dir / f"batch_{i//batch_size}.json"
                with open(batch_file, 'w') as f:
                    json.dump(current_batch, f, indent=2)
                current_batch = []
            
            # Show progress
            if (i + 1) % save_interval == 0:
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / (i + 1)
                remaining = avg_time * (len(image_files) - (i + 1))
                
                logger.info(f"\nProgress: {i + 1}/{len(image_files)} images")
                logger.info(f"Success rate: {results['successful']/(i + 1)*100:.1f}%")
                logger.info(f"Average time per image: {avg_time:.2f}s")
                logger.info(f"Estimated time remaining: {remaining/60:.1f} minutes")
        
        # Save final batch if any remaining
        if current_batch:
            batch_file = output_dir / f"batch_{len(image_files)//batch_size}.json"
            with open(batch_file, 'w') as f:
                json.dump(current_batch, f, indent=2)
        
        # Save summary
        results['end_time'] = datetime.now().isoformat()
        results['total_time'] = time.time() - start_time
        results['average_time'] = results['total_time'] / results['successful'] if results['successful'] > 0 else 0
        
        summary_file = output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate and save report
        report = generate_processing_report(results, output_dir)
        report_file = output_dir / "processing_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info("\nBatch processing completed!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Success rate: {results['successful']/len(image_files)*100:.1f}%")
        logger.info(f"Total processing time: {results['total_time']/60:.1f} minutes")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise

def generate_processing_report(results: dict, output_dir: Path) -> str:
    """
    Generate a detailed processing report.
    
    Args:
        results (dict): Processing results
        output_dir (Path): Output directory
    
    Returns:
        str: Formatted report
    """
    report = []
    report.append("=" * 80)
    report.append("BATCH PROCESSING REPORT")
    report.append("=" * 80)
    report.append(f"\nProcessing Period: {results['start_time']} to {results['end_time']}")
    report.append(f"\nTotal Images: {results['total_images']}")
    report.append(f"Successfully Processed: {results['successful']}")
    report.append(f"Failed: {results['failed_count']}")
    report.append(f"Success Rate: {results['successful']/results['total_images']*100:.1f}%")
    report.append(f"\nTotal Processing Time: {results['total_time']/60:.1f} minutes")
    report.append(f"Average Time per Image: {results['average_time']:.2f} seconds")
    
    if results['failed']:
        report.append("\nFailed Images:")
        for failure in results['failed']:
            report.append(f"- {failure['image_path']}")
            report.append(f"  Error: {failure['error']}")
            report.append(f"  Time: {failure['timestamp']}")
    
    report.append("\nOutput Directory: " + str(output_dir))
    report.append("=" * 80)
    
    return "\n".join(report)

# %%
# Test batch processing
test_images_dir = ROOT_DIR / "test_images"
if test_images_dir.exists():
    batch_results = process_batch(
        str(test_images_dir),
        detection_model,
        preprocessing_pipeline,
        postprocessing_pipeline,
        batch_size=5,
        save_interval=2
    )
else:
    logger.warning(f"Test images directory not found at {test_images_dir}")
    logger.info("Please create a test_images directory with some sample images.")

# %% [markdown]
"""
## Results Analysis
### Analysis Functions
"""

# %%
def calculate_cer(predicted: str, ground_truth: str) -> float:
    """
    Calculate Character Error Rate (CER) between predicted and ground truth text.
    
    Args:
        predicted (str): Predicted text
        ground_truth (str): Ground truth text
    
    Returns:
        float: Character Error Rate (0.0 to 1.0)
    """
    try:
        from Levenshtein import distance
        
        # Clean and normalize both texts
        predicted = ''.join(c.lower() for c in predicted if c.isalnum())
        ground_truth = ''.join(c.lower() for c in ground_truth if c.isalnum())
        
        if not ground_truth:
            return 1.0 if predicted else 0.0
        
        # Calculate Levenshtein distance
        distance = distance(predicted, ground_truth)
        
        # Calculate CER
        cer = distance / len(ground_truth)
        return min(cer, 1.0)  # Cap at 1.0
        
    except Exception as e:
        logger.error(f"Error calculating CER: {e}")
        return 1.0

def categorize_errors(predicted: str, ground_truth: str) -> dict:
    """
    Categorize errors in predicted text compared to ground truth.
    
    Args:
        predicted (str): Predicted text
        ground_truth (str): Ground truth text
    
    Returns:
        dict: Error categories and counts
    """
    try:
        from Levenshtein import editops
        
        # Clean and normalize both texts
        predicted = ''.join(c.lower() for c in predicted if c.isalnum())
        ground_truth = ''.join(c.lower() for c in ground_truth if c.isalnum())
        
        # Get edit operations
        operations = editops(ground_truth, predicted)
        
        # Count operations by type
        error_categories = {
            'insertions': 0,
            'deletions': 0,
            'substitutions': 0
        }
        
        for op in operations:
            if op[0] == 'insert':
                error_categories['insertions'] += 1
            elif op[0] == 'delete':
                error_categories['deletions'] += 1
            elif op[0] == 'replace':
                error_categories['substitutions'] += 1
        
        return error_categories
        
    except Exception as e:
        logger.error(f"Error categorizing errors: {e}")
        return {'insertions': 0, 'deletions': 0, 'substitutions': 0}

def analyze_results(
    results_dir: str,
    ground_truth_file: str = None,
    output_dir: str = None
) -> dict:
    """
    Analyze OCR results and compare with ground truth if available.
    
    Args:
        results_dir (str): Directory containing OCR results
        ground_truth_file (str): Path to ground truth JSON file
        output_dir (str): Directory to save analysis results
    
    Returns:
        dict: Analysis results
    """
    try:
        # Set up output directory
        if output_dir is None:
            output_dir = ROOT_DIR / "results" / "analysis"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load OCR results
        results = []
        for batch_file in Path(results_dir).glob("batch_*.json"):
            with open(batch_file, 'r') as f:
                results.extend(json.load(f))
        
        # Initialize analysis results
        analysis = {
            'total_images': len(results),
            'field_accuracy': {},
            'error_categories': {
                'work_order': {'insertions': 0, 'deletions': 0, 'substitutions': 0},
                'total_cost': {'insertions': 0, 'deletions': 0, 'substitutions': 0}
            },
            'cer': {
                'work_order': [],
                'total_cost': []
            },
            'confidence_scores': {
                'work_order': [],
                'total_cost': []
            },
            'processing_times': [],
            'field_detection_rates': {
                'work_order': 0,
                'total_cost': 0
            }
        }
        
        # Load ground truth if available
        ground_truth = {}
        if ground_truth_file and Path(ground_truth_file).exists():
            with open(ground_truth_file, 'r') as f:
                ground_truth = json.load(f)
        
        # Analyze each result
        for result in results:
            image_path = result['image_path']
            ocr_result = result['result']
            
            # Track processing time
            if 'processing_time' in result:
                analysis['processing_times'].append(result['processing_time'])
            
            # Get ground truth for this image if available
            gt = ground_truth.get(image_path, {})
            
            # Analyze work order
            if 'work_order' in ocr_result:
                pred_wo = str(ocr_result['work_order'])
                gt_wo = str(gt.get('work_order', ''))
                
                # Calculate CER
                cer = calculate_cer(pred_wo, gt_wo)
                analysis['cer']['work_order'].append(cer)
                
                # Categorize errors
                errors = categorize_errors(pred_wo, gt_wo)
                for category, count in errors.items():
                    analysis['error_categories']['work_order'][category] += count
                
                # Track confidence
                if 'work_order_confidence' in ocr_result:
                    analysis['confidence_scores']['work_order'].append(
                        ocr_result['work_order_confidence']
                    )
                
                # Update detection rate
                analysis['field_detection_rates']['work_order'] += 1
            
            # Analyze total cost
            if 'total_cost' in ocr_result:
                pred_cost = str(ocr_result['total_cost'])
                gt_cost = str(gt.get('total_cost', ''))
                
                # Calculate CER
                cer = calculate_cer(pred_cost, gt_cost)
                analysis['cer']['total_cost'].append(cer)
                
                # Categorize errors
                errors = categorize_errors(pred_cost, gt_cost)
                for category, count in errors.items():
                    analysis['error_categories']['total_cost'][category] += count
                
                # Track confidence
                if 'total_cost_confidence' in ocr_result:
                    analysis['confidence_scores']['total_cost'].append(
                        ocr_result['total_cost_confidence']
                    )
                
                # Update detection rate
                analysis['field_detection_rates']['total_cost'] += 1
        
        # Calculate average metrics
        for field in ['work_order', 'total_cost']:
            if analysis['cer'][field]:
                analysis['field_accuracy'][field] = {
                    'average_cer': sum(analysis['cer'][field]) / len(analysis['cer'][field]),
                    'error_categories': analysis['error_categories'][field],
                    'average_confidence': (
                        sum(analysis['confidence_scores'][field]) / 
                        len(analysis['confidence_scores'][field])
                        if analysis['confidence_scores'][field] else 0
                    ),
                    'detection_rate': (
                        analysis['field_detection_rates'][field] / 
                        analysis['total_images'] * 100
                    )
                }
        
        # Calculate average processing time
        if analysis['processing_times']:
            analysis['average_processing_time'] = sum(analysis['processing_times']) / len(analysis['processing_times'])
        
        # Save analysis results
        analysis_file = output_dir / "analysis_results.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate and save report
        report = generate_analysis_report(analysis)
        report_file = output_dir / "analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Generate visualizations
        generate_analysis_visualizations(analysis, output_dir)
        
        logger.info("\nAnalysis completed!")
        logger.info(f"Results saved to: {output_dir}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
        raise

def generate_analysis_visualizations(analysis: dict, output_dir: Path) -> None:
    """
    Generate visualizations for analysis results.
    
    Args:
        analysis (dict): Analysis results
        output_dir (Path): Directory to save visualizations
    """
    try:
        # Create figures directory
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # 1. CER Distribution
        plt.figure(figsize=(10, 6))
        for field in ['work_order', 'total_cost']:
            if analysis['cer'][field]:
                plt.hist(analysis['cer'][field], alpha=0.5, label=field)
        plt.xlabel('Character Error Rate')
        plt.ylabel('Count')
        plt.title('CER Distribution by Field')
        plt.legend()
        plt.savefig(figures_dir / "cer_distribution.png")
        plt.close()
        
        # 2. Error Categories
        for field in ['work_order', 'total_cost']:
            if field in analysis['field_accuracy']:
                plt.figure(figsize=(8, 6))
                errors = analysis['field_accuracy'][field]['error_categories']
                plt.bar(errors.keys(), errors.values())
                plt.title(f'Error Categories - {field}')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(figures_dir / f"error_categories_{field}.png")
                plt.close()
        
        # 3. Confidence vs CER
        for field in ['work_order', 'total_cost']:
            if analysis['cer'][field] and analysis['confidence_scores'][field]:
                plt.figure(figsize=(8, 6))
                plt.scatter(analysis['confidence_scores'][field], analysis['cer'][field])
                plt.xlabel('Confidence Score')
                plt.ylabel('Character Error Rate')
                plt.title(f'Confidence vs CER - {field}')
                plt.savefig(figures_dir / f"confidence_vs_cer_{field}.png")
                plt.close()
        
        # 4. Processing Time Distribution
        if analysis['processing_times']:
            plt.figure(figsize=(10, 6))
            plt.hist(analysis['processing_times'], bins=20)
            plt.xlabel('Processing Time (seconds)')
            plt.ylabel('Count')
            plt.title('Processing Time Distribution')
            plt.savefig(figures_dir / "processing_time_distribution.png")
            plt.close()
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

def generate_analysis_report(analysis: dict) -> str:
    """
    Generate a detailed analysis report.
    
    Args:
        analysis (dict): Analysis results
    
    Returns:
        str: Formatted report
    """
    report = []
    report.append("=" * 80)
    report.append("OCR RESULTS ANALYSIS REPORT")
    report.append("=" * 80)
    
    report.append(f"\nTotal Images Processed: {analysis['total_images']}")
    if 'average_processing_time' in analysis:
        report.append(f"Average Processing Time: {analysis['average_processing_time']:.2f} seconds")
    
    for field, metrics in analysis['field_accuracy'].items():
        report.append(f"\n{field.upper()} ANALYSIS:")
        report.append("-" * 40)
        report.append(f"Detection Rate: {metrics['detection_rate']:.1f}%")
        report.append(f"Average CER: {metrics['average_cer']:.4f}")
        report.append(f"Average Confidence: {metrics['average_confidence']:.4f}")
        
        report.append("\nError Categories:")
        for category, count in metrics['error_categories'].items():
            report.append(f"- {category}: {count}")
    
    report.append("\nRECOMMENDATIONS:")
    report.append("-" * 40)
    
    # Generate recommendations based on analysis
    for field, metrics in analysis['field_accuracy'].items():
        if metrics['average_cer'] > 0.1:
            report.append(f"\n{field}:")
            if metrics['error_categories']['insertions'] > metrics['error_categories']['deletions']:
                report.append("- Consider adjusting detection model to reduce false positives")
            if metrics['error_categories']['substitutions'] > 0:
                report.append("- Review recognition model for common substitution errors")
            if metrics['average_confidence'] < 0.8:
                report.append("- Low confidence scores indicate potential model improvements needed")
            if metrics['detection_rate'] < 90:
                report.append("- Low detection rate suggests potential issues with field detection")
    
    report.append("\nVISUALIZATIONS:")
    report.append("-" * 40)
    report.append("See the 'figures' directory for detailed visualizations:")
    report.append("- CER Distribution")
    report.append("- Error Categories by Field")
    report.append("- Confidence vs CER Scatter Plots")
    report.append("- Processing Time Distribution")
    
    report.append("\n" + "=" * 80)
    return "\n".join(report)

# %%
# Generate outputs
if analysis_results_dir.exists():
    # Create analysis directory if it doesn't exist
    analysis_dir = ROOT_DIR / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Save analysis results to analysis directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = analysis_dir / f"analysis_{timestamp}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Save test results to results directory
    test_results_file = ROOT_DIR / "results" / f"test_results_{timestamp}.json"
    with open(test_results_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': timestamp,
                'model_info': {
                    'detection_model': detection_model.__class__.__name__,
                    'recognition_model': recognition_model.__class__.__name__,
                    'device': str(device)
                }
            },
            'results': analysis_results
        }, f, indent=2)
    
    logger.info("\nResults have been saved:")
    logger.info(f"Analysis file: {analysis_file}")
    logger.info(f"Test results file: {test_results_file}")
else:
    logger.warning(f"Analysis results directory not found at {analysis_results_dir}")
    logger.info("Please run batch processing and analysis first.") 