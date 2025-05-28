# %% [markdown]
"""
# docTR Model Evaluation Notebook

This notebook demonstrates basic usage of the docTR model for document text recognition.
It focuses on direct model usage with clear logging of outputs.
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
import time
import json
from datetime import datetime

# External dependencies
import torch
from PIL import Image

# docTR specific imports
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

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
        
        print(f"CUDA is available with {gpu_count} GPU(s)")
        print(f"Using GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        return True
    else:
        print("CUDA is not available. Running on CPU mode.")
        return False

# Check CUDA availability
check_cuda_availability()

# %% [markdown]
"""
## Model Initialization
"""

# %%
def initialize_model() -> tuple:
    """
    Initialize docTR OCR model with default configuration.
    
    Returns:
        tuple: (model, device)
    """
    try:
        print("\nInitializing docTR model...")
        
        # Initialize model with default configuration
        model = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=True
        )
        
        # Move model to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            model.det_predictor.to(device)
            model.reco_predictor.to(device)
            print(f"Model moved to {device}")
        
        print("Model initialized successfully")
        return model, device
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise

# Initialize the model
model, device = initialize_model()

# %% [markdown]
"""
## Process Single Image
"""

# %%
def process_single_image(image_path: str) -> None:
    """
    Process a single image using docTR model and display results.
    
    Args:
        image_path (str): Path to the image file
    """
    try:
        print(f"\nProcessing image: {image_path}")
        
        # Load image using DocumentFile
        print("Loading image with DocumentFile...")
        doc = DocumentFile.from_images(image_path)
        print(f"Document loaded successfully")
        
        # Run inference
        print("\nRunning inference...")
        start_time = time.time()
        with torch.no_grad():
            result = model(doc)
        processing_time = time.time() - start_time
        print(f"Inference completed in {processing_time:.2f} seconds")
        
        # Display results
        print("\n" + "="*50)
        print("MODEL OUTPUT")
        print("="*50)
        
        for page_idx, page in enumerate(result.pages):
            print(f"\nPage {page_idx + 1}:")
            print(f"Number of blocks: {len(page.blocks)}")
            
            for block_idx, block in enumerate(page.blocks):
                print(f"\nBlock {block_idx + 1}:")
                print(f"Block confidence: {block.confidence:.3f}")
                print(f"Number of lines: {len(block.lines)}")
                
                for line_idx, line in enumerate(block.lines):
                    print(f"\n  Line {line_idx + 1}:")
                    print(f"  Line confidence: {line.confidence:.3f}")
                    print(f"  Number of words: {len(line.words)}")
                    
                    for word_idx, word in enumerate(line.words):
                        print(f"    Word {word_idx + 1}:")
                        print(f"    Text: {word.value}")
                        print(f"    Confidence: {word.confidence:.3f}")
                        print(f"    Geometry: {word.geometry}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

# %% [markdown]
"""
## Test with Sample Image
"""

# %%
# Test with a sample image
test_image_path = ROOT_DIR / "data" / "images" / "1017.jpg"
if test_image_path.exists():
    process_single_image(str(test_image_path))
else:
    print(f"Test image not found at {test_image_path}")
    print("Please ensure the image file exists at the specified path.")

# %% [markdown]
"""
## Batch Processing
"""

# %%
def process_batch(image_dir: str, output_dir: str = None) -> None:
    """
    Process a batch of images and save results.
    
    Args:
        image_dir (str): Directory containing images to process
        output_dir (str): Directory to save results
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
        
        print(f"\nFound {len(image_files)} images to process")
        
        # Process each image
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\nProcessing image {i}/{len(image_files)}: {image_path.name}")
            
            try:
                # Load and process image
                doc = DocumentFile.from_images(str(image_path))
                with torch.no_grad():
                    result = model(doc)
                
                # Extract text and confidence scores
                image_result = {
                    'image_path': str(image_path),
                    'timestamp': datetime.now().isoformat(),
                    'pages': []
                }
                
                for page in result.pages:
                    page_data = {
                        'blocks': []
                    }
                    for block in page.blocks:
                        block_data = {
                            'confidence': block.confidence,
                            'lines': []
                        }
                        for line in block.lines:
                            line_data = {
                                'confidence': line.confidence,
                                'words': []
                            }
                            for word in line.words:
                                line_data['words'].append({
                                    'text': word.value,
                                    'confidence': word.confidence,
                                    'geometry': word.geometry
                                })
                            block_data['lines'].append(line_data)
                        page_data['blocks'].append(block_data)
                    image_result['pages'].append(page_data)
                
                results.append(image_result)
                print(f"Successfully processed {image_path.name}")
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue
        
        # Save results
        output_file = output_dir / "batch_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBatch processing completed!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        raise

# %%
# Test batch processing
test_images_dir = ROOT_DIR / "test_images"
if test_images_dir.exists():
    process_batch(str(test_images_dir))
else:
    print(f"Test images directory not found at {test_images_dir}")
    print("Please create a test_images directory with some sample images.") 