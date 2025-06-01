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
            pretrained=True,
            resolve_blocks=True
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
        
        # Load and display the image
        print("Loading and displaying image...")
        image = Image.open(image_path)
        
        # Create a display version of the image with a max size of 800x800
        display_image = image.copy()
        max_display_size = (800, 800)
        display_image.thumbnail(max_display_size, Image.Resampling.LANCZOS)
        
        # Display the image
        print("\nInput Image (resized for display):")
        display(display_image)
        
        # Load image using DocumentFile
        print("\nLoading image with DocumentFile...")
        doc = DocumentFile.from_images(image_path)
        print(f"Document loaded successfully")
        
        # Run inference
        print("\nRunning inference...")
        start_time = time.time()
        with torch.no_grad():
            result = model(doc)
        processing_time = time.time() - start_time
        print(f"Inference completed in {processing_time:.2f} seconds")
        
        # Add rendered text output for easy comparison
        print("\n" + "="*50)
        print("RENDERED TEXT OUTPUT")
        print("="*50)
        rendered_text = result.render()
        print(rendered_text)
        print("="*50)
        
        # Store result globally for post-processing
        global last_result
        last_result = result
        
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
## Post-Process Single Image
"""

# %%
def extract_work_order_and_total(result) -> dict:
    """
    Extract work order number and total cost from docTR result using spatial proximity and label-value relationships.
    
    Args:
        result: docTR result object
        
    Returns:
        dict: Extracted data with work_order_number and total_cost
    """
    try:
        # Convert result to JSON for easier processing
        json_result = result.export()
        
        # Debug: Print the full JSON structure
        print("DEBUG: Full JSON structure:")
        print(json.dumps(json_result, indent=2))
        print("=" * 80)
        
        extracted_data = {
            "work_order_number": None,
            "total_cost": None,
            "extraction_confidence": {
                "work_order_found": False,
                "total_cost_found": False,
                "spatial_match": False
            }
        }
        
        def get_block_info(block):
            """Extract block information including text and spatial coordinates."""
            block_words = []
            for line in block['lines']:
                for word in line['words']:
                    block_words.append(word)
            
            if not block_words:
                return None, None, None, None
            
            # Calculate block center point for better spatial comparison
            all_coords = []
            for word in block_words:
                coords = word['geometry']
                all_coords.extend(coords)
            
            if not all_coords:
                return None, None, None, None
            
            # Calculate center point
            center_x = sum(coord[0] for coord in all_coords) / len(all_coords)
            center_y = sum(coord[1] for coord in all_coords) / len(all_coords)
            
            # Get block text
            block_text = ' '.join(word['value'] for word in block_words)
            
            return block_words, block_text, center_x, center_y
        
        def find_nearby_numeric_value(target_x, target_y, all_blocks, max_distance=0.15):
            """Find numeric values near a target coordinate."""
            candidates = []
            
            for block in all_blocks:
                block_words, block_text, center_x, center_y = get_block_info(block)
                if not block_words:
                    continue
                
                # Calculate distance from target
                distance = ((center_x - target_x) ** 2 + (center_y - target_y) ** 2) ** 0.5
                
                if distance <= max_distance:
                    # Look for numeric values in this block
                    for word in block_words:
                        word_text = word['value'].strip()
                        if word_text.isdigit() and 4 <= len(word_text) <= 6:
                            candidates.append({
                                'value': word_text,
                                'distance': distance,
                                'same_line': abs(center_y - target_y) < 0.05,
                                'to_right': center_x > target_x
                            })
            
            # Sort candidates by priority: same line, to the right, then by distance
            candidates.sort(key=lambda x: (
                not x['same_line'],  # Same line first
                not x['to_right'],   # To the right second
                x['distance']        # Closest third
            ))
            
            return candidates[0]['value'] if candidates else None
        
        def find_nearby_monetary_value(target_x, target_y, all_blocks, label_block=None, max_distance=0.15):
            """Find monetary values near a target coordinate, including within the label block itself."""
            candidates = []
            
            # First, check within the label block itself if provided
            if label_block:
                block_words, block_text, center_x, center_y = get_block_info(label_block)
                print(f"DEBUG: Checking within label block: '{block_text}'")
                if block_words:
                    for word in block_words:
                        word_text = word['value'].strip()
                        print(f"DEBUG: Checking word: '{word_text}'")
                        # Improved monetary pattern - look for $ symbol OR decimal numbers that look like money
                        is_monetary = False
                        if '$' in word_text:
                            is_monetary = True
                        elif '.' in word_text:
                            # Check if it's a decimal number that could be monetary (like 950.00)
                            clean_test = word_text.replace(',', '').strip()
                            try:
                                float_val = float(clean_test)
                                # Check if it has exactly 2 decimal places (typical for money)
                                if '.' in clean_test and len(clean_test.split('.')[1]) == 2:
                                    is_monetary = True
                                # Or if it's a reasonable monetary amount (> 1.00)
                                elif float_val >= 1.0:
                                    is_monetary = True
                            except ValueError:
                                pass
                        
                        if is_monetary:
                            clean_amount = word_text.replace('$', '').replace(',', '').strip()
                            print(f"DEBUG: Found potential monetary value: '{clean_amount}'")
                            try:
                                float(clean_amount)
                                print(f"DEBUG: Successfully validated monetary value: '{clean_amount}'")
                                candidates.append({
                                    'value': clean_amount,
                                    'distance': 0,  # Same block = distance 0
                                    'same_line': True,  # Same block = same line
                                    'to_right': True   # Assume to the right within block
                                })
                            except ValueError:
                                print(f"DEBUG: Failed to validate monetary value: '{clean_amount}'")
                                continue
            
            # Then check nearby blocks
            for block in all_blocks:
                block_words, block_text, center_x, center_y = get_block_info(block)
                if not block_words:
                    continue
                
                # Skip the label block if we already checked it
                if label_block and block is label_block:
                    continue
                
                # Calculate distance from target
                distance = ((center_x - target_x) ** 2 + (center_y - target_y) ** 2) ** 0.5
                
                if distance <= max_distance:
                    # Look for monetary values in this block
                    for word in block_words:
                        word_text = word['value'].strip()
                        # Improved monetary pattern - look for $ symbol OR decimal numbers that look like money
                        is_monetary = False
                        if '$' in word_text:
                            is_monetary = True
                        elif '.' in word_text:
                            # Check if it's a decimal number that could be monetary (like 950.00)
                            clean_test = word_text.replace(',', '').strip()
                            try:
                                float_val = float(clean_test)
                                # Check if it has exactly 2 decimal places (typical for money)
                                if '.' in clean_test and len(clean_test.split('.')[1]) == 2:
                                    is_monetary = True
                                # Or if it's a reasonable monetary amount (> 1.00)
                                elif float_val >= 1.0:
                                    is_monetary = True
                            except ValueError:
                                pass
                        
                        if is_monetary:
                            clean_amount = word_text.replace('$', '').replace(',', '').strip()
                            try:
                                float(clean_amount)
                                candidates.append({
                                    'value': clean_amount,
                                    'distance': distance,
                                    'same_line': abs(center_y - target_y) < 0.05,
                                    'to_right': center_x > target_x
                                })
                            except ValueError:
                                continue
            
            # Sort candidates by priority
            candidates.sort(key=lambda x: (
                not x['same_line'],
                not x['to_right'],
                x['distance']
            ))
            
            return candidates[0]['value'] if candidates else None
        
        # Process each page (usually just one)
        for page in json_result['pages']:
            all_blocks = page['blocks']
            
            # Find work order number using label-value proximity
            for block in all_blocks:
                block_words, block_text, center_x, center_y = get_block_info(block)
                if not block_words:
                    continue
                
                block_text_lower = block_text.lower()
                print(f"DEBUG: Checking block: '{block_text}' -> '{block_text_lower}'")
                
                # Look for MJM Order Number label
                if ('mjm' in block_text_lower and 'order' in block_text_lower and 'number' in block_text_lower):
                    extracted_data["extraction_confidence"]["spatial_match"] = True
                    
                    # Find nearby numeric values
                    work_order = find_nearby_numeric_value(center_x, center_y, all_blocks)
                    if work_order:
                        extracted_data["work_order_number"] = work_order
                        extracted_data["extraction_confidence"]["work_order_found"] = True
                
                # Look for Grand Total label (separate condition, not elif!)
                if ('grand' in block_text_lower and 'total' in block_text_lower):
                    print(f"DEBUG: Found Grand Total block: '{block_text}'")
                    extracted_data["extraction_confidence"]["spatial_match"] = True
                    
                    # Find nearby monetary values, including within this same block
                    total_cost = find_nearby_monetary_value(center_x, center_y, all_blocks, label_block=block)
                    print(f"DEBUG: Total cost found: {total_cost}")
                    if total_cost:
                        extracted_data["total_cost"] = total_cost
                        extracted_data["extraction_confidence"]["total_cost_found"] = True
        
        return extracted_data
        
    except Exception as e:
        print(f"Error in post-processing: {e}")
        return {
            "work_order_number": None,
            "total_cost": None,
            "extraction_confidence": {
                "work_order_found": False,
                "total_cost_found": False,
                "spatial_match": False
            },
            "error": str(e)
        }

def post_process_single_image():
    """
    Post-process the last processed image result to extract structured data.
    """
    try:
        # Check if we have a result to process
        if 'last_result' not in globals():
            print("No image result available. Please run the single image test first.")
            return
        
        print("\n" + "="*50)
        print("POST-PROCESSING EXTRACTION")
        print("="*50)
        
        # Extract structured data
        extracted_data = extract_work_order_and_total(last_result)
        
        # Display results
        print("\nExtracted Data:")
        print(json.dumps(extracted_data, indent=2))
        
        # Provide user feedback
        if extracted_data["extraction_confidence"]["work_order_found"]:
            print(f"\n✅ Work Order Number found: {extracted_data['work_order_number']}")
        else:
            print("\n❌ Work Order Number not found")
            
        if extracted_data["extraction_confidence"]["total_cost_found"]:
            print(f"✅ Total Cost found: ${extracted_data['total_cost']}")
        else:
            print("❌ Total Cost not found")
            
        if extracted_data["extraction_confidence"]["spatial_match"]:
            print("✅ Spatial filtering successful")
        else:
            print("❌ No spatial matches found")
        
        return extracted_data
        
    except Exception as e:
        print(f"Error in post-processing: {e}")
        return None

# Run post-processing on the last result
post_process_single_image()



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