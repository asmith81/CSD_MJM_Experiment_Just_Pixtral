# %% [markdown]
"""
# Pixtral Model Evaluation Notebook

This notebook evaluates the Pixtral-12B model's performance on invoice data extraction.
It follows the project's notebook handling rules and functional programming approach.
"""

# %% [markdown]
"""
## Setup and Configuration
### Initial Imports
"""

# %%
import os
import sys
import subprocess
from pathlib import Path
import logging
import json
from datetime import datetime
import torch
from PIL import Image
from typing import Union, Dict, Any, List, Literal
import yaml

# %% [markdown]
"""
### Define a Function and Global Variable to Decide and Hold the Prompt
"""

# %%
# Global variable to store selected prompt
SELECTED_PROMPT = None

def load_prompt_files() -> Dict[str, Dict]:
    """Load all prompt YAML files from the config/prompts directory."""
    prompts_dir = Path("config/prompts")
    prompt_files = {
        "basic_extraction": prompts_dir / "basic_extraction.yaml",
        "detailed": prompts_dir / "detailed.yaml",
        "few_shot": prompts_dir / "few_shot.yaml",
        "locational": prompts_dir / "locational.yaml",
        "step_by_step": prompts_dir / "step_by_step.yaml"
    }
    
    loaded_prompts = {}
    for name, file_path in prompt_files.items():
        with open(file_path, 'r') as f:
            loaded_prompts[name] = yaml.safe_load(f)
    return loaded_prompts

def select_prompt() -> str:
    """Allow user to select a prompt type and return the prompt text."""
    global SELECTED_PROMPT
    
    prompts = load_prompt_files()
    print("\nAvailable prompt types:")
    for i, name in enumerate(prompts.keys(), 1):
        print(f"{i}. {name.replace('_', ' ').title()}")
    
    while True:
        try:
            choice = int(input("\nSelect a prompt type (1-5): "))
            if 1 <= choice <= len(prompts):
                selected_name = list(prompts.keys())[choice - 1]
                SELECTED_PROMPT = prompts[selected_name]
                print(f"\nSelected prompt type: {selected_name.replace('_', ' ').title()}")
                print("\nPrompt text:")
                print("-" * 50)
                print(SELECTED_PROMPT['prompts'][0]['text'])
                print("-" * 50)
                return selected_name
            else:
                print("Invalid choice. Please select a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")
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
    if not (ROOT_DIR / "pixtral_model.py").exists() or not (ROOT_DIR / "requirements_pixtral.txt").exists():
        raise RuntimeError("Could not find both pixtral_model.py and requirements_pixtral.txt in the same directory")
except NameError:
    # When running in a notebook, look for the files in current directory
    current_dir = Path.cwd()
    if not (current_dir / "pixtral_model.py").exists() or not (current_dir / "requirements_pixtral.txt").exists():
        raise RuntimeError("Could not find both pixtral_model.py and requirements_pixtral.txt in the current directory")
    ROOT_DIR = current_dir

sys.path.append(str(ROOT_DIR))

# Create results directory
results_dir = ROOT_DIR / "results"
results_dir.mkdir(exist_ok=True)
logger.info(f"Results will be saved to: {results_dir}")

# %% [markdown]
"""
## Install Dependencies
"""

# %%
!pip install tqdm
from tqdm import tqdm


def install_dependencies():
    """Install required dependencies with progress tracking."""
    steps = [
        ("Base requirements", [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT_DIR / "requirements_pixtral.txt")]),
        ("PyTorch", [
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.1.0",
            "torchvision==0.16.0",
            "torchaudio==2.1.0",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
    ]
    
    for step_name, command in tqdm(steps, desc="Installing dependencies"):
        try:
            subprocess.check_call(command)
            logger.info(f"Successfully installed {step_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing {step_name}: {e}")
            raise

# Install dependencies
install_dependencies()

# %% [markdown]
"""
## Memory Resource Check
"""

# %%
def check_memory_resources():
    """
    Check available GPU memory, system RAM, and compare with Pixtral model requirements.
    Returns a dictionary with memory information and recommendations.
    """
    memory_info = {
        "gpu_available": False,
        "gpu_memory": None,
        "system_ram": None,
        "recommendations": []
    }
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        memory_info["gpu_available"] = True
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        memory_info["gpu_memory"] = round(gpu_mem, 2)
        logger.info(f"GPU Memory Available: {memory_info['gpu_memory']} GB")
    else:
        logger.warning("No GPU available. This will significantly impact model performance.")
        memory_info["recommendations"].append("No GPU detected. Consider using a GPU-enabled environment.")
    
    # Check system RAM
    import psutil
    system_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
    memory_info["system_ram"] = round(system_ram, 2)
    logger.info(f"System RAM Available: {memory_info['system_ram']} GB")
    
    # Model requirements and recommendations
    model_requirements = {
        "no_quantization": 93.0,  # GB
        "8bit_quantization": 46.0,  # GB
        "4bit_quantization": 23.0   # GB
    }
    
    # Add recommendations based on available resources
    if memory_info["gpu_available"]:
        if memory_info["gpu_memory"] >= model_requirements["no_quantization"]:
            memory_info["recommendations"].append("Sufficient GPU memory for full precision model")
        elif memory_info["gpu_memory"] >= model_requirements["8bit_quantization"]:
            memory_info["recommendations"].append("Consider using 8-bit quantization")
        elif memory_info["gpu_memory"] >= model_requirements["4bit_quantization"]:
            memory_info["recommendations"].append("Consider using 4-bit quantization")
        else:
            memory_info["recommendations"].append("Insufficient GPU memory. Consider using CPU offloading or a different model.")
    
    # Check if system RAM is sufficient for CPU offloading if needed
    if memory_info["system_ram"] < model_requirements["4bit_quantization"]:
        memory_info["recommendations"].append("Warning: System RAM may be insufficient for CPU offloading")
    
    return memory_info

# Check memory resources
memory_status = check_memory_resources()
logger.info("Memory Status:")
for key, value in memory_status.items():
    if key != "recommendations":
        logger.info(f"{key}: {value}")
logger.info("Recommendations:")
for rec in memory_status["recommendations"]:
    logger.info(f"- {rec}")

# %% [markdown]
"""
## Quantization Selection
"""

# %%
def select_quantization() -> Literal["bfloat16", "int8", "int4"]:
    """
    Select quantization level for the model.
    Returns one of: "bfloat16", "int8", "int4"
    """
    print("\nAvailable quantization options:")
    print("1. bfloat16 (full precision, 93GB VRAM)")
    print("2. int8 (8-bit, 46GB VRAM)")
    print("3. int4 (4-bit, 23GB VRAM)")
    
    while True:
        try:
            choice = int(input("\nSelect quantization (1-3): "))
            if choice == 1:
                return "bfloat16"
            elif choice == 2:
                return "int8"
            elif choice == 3:
                return "int4"
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
        except ValueError:
            print("Please enter a number between 1 and 3.")

# Select quantization
quantization = select_quantization()
logger.info(f"Selected quantization: {quantization}")

# %% [markdown]
"""
## Flash Attention Configuration
"""

# %%
def configure_flash_attention() -> bool:
    """
    Check if Flash Attention is available and configure it.
    Returns True if Flash Attention is enabled, False otherwise.
    """
    try:
        import flash_attn
        
        # Check if GPU supports Flash Attention
        if not torch.cuda.is_available():
            logger.warning("Flash Attention requires CUDA GPU. Disabling Flash Attention.")
            return False
            
        # Get GPU compute capability
        major, minor = torch.cuda.get_device_capability()
        compute_capability = float(f"{major}.{minor}")
        
        # Flash Attention 2 requires compute capability >= 8.0
        if compute_capability >= 8.0:
            logger.info("Flash Attention 2 enabled - GPU supports compute capability 8.0+")
            return True
        else:
            logger.warning(f"GPU compute capability {compute_capability} does not support Flash Attention 2")
            return False
            
    except ImportError:
        logger.warning("Flash Attention not installed. Please install with: pip install flash-attn")
        return False

# Configure Flash Attention
use_flash_attention = configure_flash_attention()
logger.info(f"Flash Attention Status: {'Enabled' if use_flash_attention else 'Disabled'}")

# %% [markdown]
"""
## Device Mapping Configuration
"""

# %%
def configure_device_mapping() -> dict:
    """
    Configure device mapping for GPU.
    Returns device map configuration for model loading.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. This notebook requires a GPU to run.")
    
    # Use single GPU setup
    device_map = {"": 0}
    logger.info("Using single GPU setup")
    return device_map

# Configure device mapping
device_map = configure_device_mapping()
logger.info(f"Device Map: {device_map}")

# %% [markdown]
"""
## Optional: Multi-GPU Configuration
"""

# %%
def configure_multi_gpu() -> dict:
    """
    Configure device mapping for multiple GPUs.
    Returns device map configuration for parallel processing.
    """
    if torch.cuda.device_count() <= 1:
        logger.info("Only one GPU detected. Using single GPU configuration.")
        return {"": 0}
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} GPUs. Configuring for parallel processing.")
    
    # Create balanced device map
    device_map = {}
    for i in range(num_gpus):
        device_map[f"model.layers.{i}"] = i % num_gpus
    
    # Map remaining layers to first GPU
    device_map[""] = 0
    
    return device_map

# Optional: Use this instead of single GPU configuration if multiple GPUs are available
# device_map = configure_multi_gpu()
# logger.info(f"Multi-GPU Device Map: {device_map}")

# %% [markdown]
"""
## Version Checks
"""

# %%
def check_versions():
    """
    Check required package versions for Pixtral model.
    Logs any version mismatches and raises error if critical.
    """
    import pkg_resources
    
    # Required versions from requirements.txt and model card
    required_versions = {
        "transformers": "4.50.3",  # Must be >=4.45
        "Pillow": "9.3.0",
        "torch": "2.1.0",
        "accelerate": "0.26.0",
        "bitsandbytes": "0.45.5",
        "flash-attn": "2.5.0"
    }
    
    version_issues = []
    for package, required_version in required_versions.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if package == "transformers" and pkg_resources.parse_version(installed_version) < pkg_resources.parse_version("4.45.0"):
                version_issues.append(f"transformers version {installed_version} is below minimum required version 4.45.0")
            elif pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(required_version):
                version_issues.append(f"{package} version {installed_version} is below required version {required_version}")
        except pkg_resources.DistributionNotFound:
            version_issues.append(f"{package} is not installed")
    
    if version_issues:
        for issue in version_issues:
            logger.warning(issue)
        if any("transformers" in issue for issue in version_issues):
            raise ImportError("transformers version must be >=4.45.0 for Pixtral model")
    else:
        logger.info("All package versions meet requirements")

# Check versions
check_versions()

# %% [markdown]
"""
## Model Download
"""

# %%
def download_pixtral_model(model_id: str = "mistral-community/pixtral-12b", 
                         max_retries: int = 2,
                         retry_delay: int = 5) -> tuple:
    """
    Download the Pixtral model with retry logic and memory monitoring.
    
    Args:
        model_id: HuggingFace model ID
        max_retries: Maximum number of download attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        tuple: (model, processor) if successful
        
    Raises:
        RuntimeError: If download fails after max retries
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
    import time
    import psutil
    
    def log_memory_usage(stage: str):
        """Log current memory usage"""
        gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        ram = psutil.virtual_memory().used / (1024**3)
        logger.info(f"Memory usage at {stage}: GPU={gpu_mem:.2f}GB, RAM={ram:.2f}GB")
    
    # Log initial memory usage
    log_memory_usage("start")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}")
            
            # Configure model loading based on selected quantization
            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": True
            }
            
            if quantization == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif quantization == "int8":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            elif quantization == "int4":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Download model and processor
            model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
            processor = AutoProcessor.from_pretrained(model_id)
            
            # Log final memory usage
            log_memory_usage("complete")
            
            logger.info("Model and processor downloaded successfully")
            return model, processor
            
        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to download model after {max_retries} attempts: {str(e)}")

model, processor = download_pixtral_model()
logger.info("Model and processor ready for use")

# %% [markdown]
"""
## Initialize Model
"""

# %%
def initialize_model(quantization: Literal["bfloat16", "int8", "int4"]) -> tuple:
    """
    Initialize the Pixtral model and processor with the specified quantization.
    
    Args:
        quantization: The quantization level to use ("bfloat16", "int8", or "int4")
        
    Returns:
        tuple: (model, processor) if successful
        
    Raises:
        RuntimeError: If model initialization fails
    """
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
        
        # Configure model loading based on selected quantization
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": True
        }
        
        if quantization == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif quantization == "int8":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        elif quantization == "int4":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Initialize model and processor
        model = LlavaForConditionalGeneration.from_pretrained("mistral-community/pixtral-12b", **model_kwargs)
        processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")
        
        return model, processor
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {str(e)}")

# Initialize model and processor
model, processor = initialize_model(quantization)
logger.info(f"Model and processor initialized with {quantization} quantization")

# %% [markdown]
"""
## Prompt Selection
Select a prompt type for the model evaluation.
"""

# %%
# Run the prompt selection
selected_prompt_type = select_prompt()
logger.info(f"Selected prompt type: {selected_prompt_type}")

# The selected prompt is now stored in the global variable SELECTED_PROMPT
# This can be accessed in subsequent cells for model evaluation

# %% [markdown]
"""
## Single Image Test
Run the model on a single image using the selected prompt.
"""

# %%
def format_prompt(prompt_text: str) -> str:
    """Format the prompt using the Pixtral template."""
    config = yaml.safe_load(open("config/pixtral.yaml", 'r'))
    special_tokens = config['model_params']['special_tokens']
    
    # Format the prompt with special tokens and image token
    formatted_prompt = f"{special_tokens[2]}\n{prompt_text}\n{special_tokens[0]}\n{special_tokens[1]}\n{special_tokens[3]}"
    return formatted_prompt

def load_and_process_image(image_path: str) -> Image.Image:
    """Load and process the image according to Pixtral specifications."""
    config = yaml.safe_load(open("config/pixtral.yaml", 'r'))
    
    # Load image
    image = Image.open(image_path)
    
    # Convert to RGB if needed
    if image.mode != config['model_params']['image_format']:
        image = image.convert(config['model_params']['image_format'])
    
    # Resize if needed
    max_size = config['model_params']['max_image_size']
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    return image

def run_single_image_test():
    """Run the model on a single image with the selected prompt."""
    # Get the first .jpg file from data/images
    image_dir = Path("data/images")
    image_files = list(image_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError("No .jpg files found in data/images directory")
    
    image_path = str(image_files[0])
    
    # Load and process image
    image = load_and_process_image(image_path)
    
    # Create a display version of the image with a max size of 800x800
    display_image = image.copy()
    max_display_size = (800, 800)
    display_image.thumbnail(max_display_size, Image.Resampling.LANCZOS)
    
    # Format the prompt
    prompt_text = SELECTED_PROMPT['prompts'][0]['text']
    formatted_prompt = format_prompt(prompt_text)
    
    # Display the image
    print("\nInput Image (resized for display):")
    display(display_image)
    
    # Display the prompt
    print("\nFormatted Prompt:")
    print("-" * 50)
    print(formatted_prompt)
    print("-" * 50)
    
    # Prepare model inputs using the original image
    inputs = processor(
        text=formatted_prompt,
        images=[image],  # Pass image as a list
        return_tensors="pt"
    )
    
    # Move inputs to the correct device and dtype
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Convert inputs to the correct dtype based on quantization
    if quantization == "bfloat16":
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
    elif quantization in ["int8", "int4"]:
        # For quantized models, convert to float16
        inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
    
    # Get inference parameters from config
    config = yaml.safe_load(open("config/pixtral.yaml", 'r'))
    inference_params = config['inference']
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=inference_params['max_new_tokens'],
            do_sample=inference_params['do_sample'],
            temperature=inference_params['temperature'],
            top_k=inference_params['top_k'],
            top_p=inference_params['top_p']
        )
    
    # Decode and display response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    print("\nModel Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)

# Run the single image test
try:
    run_single_image_test()
except Exception as e:
    logger.error(f"Error during single image test: {str(e)}")
    raise

# %% [markdown]
"""
## Batch Test
Run the model on all images and save results.
"""

# %%

def generate_test_id() -> str:
    """Generate a unique test identifier using timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def collect_test_metadata() -> dict:
    """Collect metadata about the current test configuration."""
    return {
        "test_id": generate_test_id(),
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "name": "Pixtral-12B",
            "version": "1.0",
            "model_id": "mistral-community/pixtral-12b",
            "quantization": quantization,
            "parameters": {
                "use_flash_attention": use_flash_attention,
                "device_map": device_map
            }
        },
        "prompt_type": selected_prompt_type,
        "system_resources": check_memory_resources()
    }

def get_most_recent_test_results() -> Path:
    """Get the most recent test results file from the results directory."""
    results_dir = Path("results")
    if not results_dir.exists():
        raise FileNotFoundError("Results directory not found")
    
    # Get all test results files
    test_files = list(results_dir.glob("test_results_*.json"))
    if not test_files:
        raise FileNotFoundError("No test results files found")
    
    # Sort by modification time (most recent first)
    test_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return test_files[0]

def select_test_results_file() -> Path:
    """Allow user to select a test results file or use the most recent."""
    try:
        most_recent = get_most_recent_test_results()
        print(f"\nMost recent test results file: {most_recent.name}")
        print("\nOptions:")
        print("1. Use most recent file")
        print("2. Select a different file")
        
        while True:
            try:
                choice = int(input("\nSelect an option (1-2): "))
                if choice == 1:
                    return most_recent
                elif choice == 2:
                    # List all available files
                    results_dir = Path("results")
                    test_files = list(results_dir.glob("test_results_*.json"))
                    test_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    print("\nAvailable test results files:")
                    for i, file in enumerate(test_files, 1):
                        print(f"{i}. {file.name}")
                    
                    file_choice = int(input("\nSelect a file number: "))
                    if 1 <= file_choice <= len(test_files):
                        return test_files[file_choice - 1]
                    else:
                        print("Invalid choice. Please select a valid file number.")
                else:
                    print("Invalid choice. Please select 1 or 2.")
            except ValueError:
                print("Please enter a valid number.")
    except FileNotFoundError as e:
        raise RuntimeError(f"Error selecting test results file: {str(e)}")

def extract_json_from_response(response: str) -> tuple[dict, str]:
    """
    Extract and parse JSON from model response.
    Returns tuple of (parsed_json, error_message).
    If successful, error_message will be empty.
    """
    try:
        # Find the second occurrence of {
        first_brace = response.find('{')
        if first_brace == -1:
            return None, "No JSON object found in response"
            
        second_brace = response.find('{', first_brace + 1)
        if second_brace == -1:
            return None, "No second JSON object found in response"
        
        # Find the matching closing brace
        brace_count = 1
        end_idx = second_brace + 1
        while brace_count > 0 and end_idx < len(response):
            if response[end_idx] == '{':
                brace_count += 1
            elif response[end_idx] == '}':
                brace_count -= 1
            end_idx += 1
            
        if brace_count != 0:
            return None, "Unmatched braces in JSON object"
            
        # Extract and parse JSON
        json_str = response[second_brace:end_idx].strip()
        parsed_json = json.loads(json_str)
        
        # Validate structure
        if not isinstance(parsed_json, dict):
            return None, "Response is not a JSON object"
        
        # Validate required fields
        required_fields = {"work_order_number", "total_cost"}
        missing_fields = required_fields - set(parsed_json.keys())
        if missing_fields:
            return None, f"Missing required fields: {missing_fields}"
        
        return parsed_json, ""
        
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def process_all_images(results_file: Path) -> list:
    """Process all images in the data/images directory and collect responses."""
    results = []
    image_dir = Path("data/images")
    image_files = list(image_dir.glob("*.jpg"))
    
    if not image_files:
        raise FileNotFoundError("No .jpg files found in data/images directory")
    
    for image_path in image_files:
        try:
            # Load and process image
            image = load_and_process_image(str(image_path))
            
            # Format the prompt
            prompt_text = SELECTED_PROMPT['prompts'][0]['text']
            formatted_prompt = format_prompt(prompt_text)
            
            # Prepare model inputs
            inputs = processor(
                text=formatted_prompt,
                images=[image],
                return_tensors="pt"
            )
            
            # Move inputs to the correct device and dtype
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Convert inputs to the correct dtype based on quantization
            if quantization == "bfloat16":
                inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
            elif quantization in ["int8", "int4"]:
                # For quantized models, convert to float16
                inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # Get inference parameters from config
            config = yaml.safe_load(open("config/pixtral.yaml", 'r'))
            inference_params = config['inference']
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=inference_params['max_new_tokens'],
                    do_sample=inference_params['do_sample'],
                    temperature=inference_params['temperature'],
                    top_k=inference_params['top_k'],
                    top_p=inference_params['top_p']
                )
            
            # Decode response
            response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract and parse JSON
            parsed_json, error_message = extract_json_from_response(response)
            
            # Create result entry
            result = {
                "image_name": image_path.name,
                "status": "completed" if parsed_json else "error",
                "timestamp": datetime.now().isoformat()
            }
            
            if parsed_json:
                result["extracted_data"] = parsed_json
            else:
                result["error"] = {
                    "message": error_message,
                    "raw_response": response
                }
            
            # Add to results
            results.append(result)
            
            # Save incremental results
            save_incremental_results(results_file, results)
            
            logger.info(f"Processed image: {image_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing image {image_path.name}: {str(e)}")
            result = {
                "image_name": image_path.name,
                "status": "error",
                "error": {
                    "message": str(e),
                    "type": "processing_error"
                },
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            # Save incremental results even on error
            save_incremental_results(results_file, results)
    
    return results

def save_incremental_results(results_file: Path, results: list):
    """Save results incrementally to a temporary file."""
    # Create temporary file if it doesn't exist
    if not results_file.exists():
        initial_data = {
            "metadata": collect_test_metadata(),
            "prompt": {
                "raw_text": SELECTED_PROMPT['prompts'][0]['text'],
                "formatted": format_prompt(SELECTED_PROMPT['prompts'][0]['text'])
            },
            "results": []
        }
        with open(results_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
    
    # Update the results section
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Update only the results section
    data["results"] = results
    
    # Save the updated data
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)

def run_batch_test():
    """Run the model on all images and save results."""
    try:
        # Generate unique filename
        test_id = generate_test_id()
        results_file = results_dir / f"test_results_{test_id}.json"
        
        # Process all images with incremental saving
        results = process_all_images(results_file)
        
        logger.info(f"Batch test completed. Results saved to: {results_file}")
        return str(results_file)
        
    except Exception as e:
        logger.error(f"Error during batch test: {str(e)}")
        raise

# Example usage:
run_batch_test()

# %% [markdown]
"""
## Analysis Functions
Functions for analyzing model performance and generating analysis reports.
"""

# %%
def normalize_total_cost(cost_str: str) -> float:
    """Convert a cost string to a float by removing currency symbols and commas."""
    if not cost_str:
        return None
    # If already a float, return as is
    if isinstance(cost_str, (int, float)):
        return float(cost_str)
    # Remove $ and commas, then convert to float
    return float(cost_str.replace('$', '').replace(',', '').strip())

def categorize_work_order_error(predicted: str, ground_truth: str) -> str:
    """Categorize the type of error in work order number prediction."""
    if not predicted or not ground_truth:
        return "No Extraction"
    if predicted == ground_truth:
        return "Exact Match"
    # Check if prediction looks like a date (contains - or /)
    if '-' in predicted or '/' in predicted:
        return "Date Confusion"
    # Check for partial match (some digits match)
    if any(digit in ground_truth for digit in predicted):
        return "Partial Match"
    return "Completely Wrong"

def categorize_total_cost_error(predicted: float, ground_truth: float) -> str:
    """Categorize the type of error in total cost prediction."""
    if predicted is None or ground_truth is None:
        return "No Extraction"
    if predicted == ground_truth:
        return "Numeric Match"
    
    # Convert to strings for digit comparison
    pred_str = str(int(predicted))
    truth_str = str(int(ground_truth))
    
    # Check for digit reversal
    if pred_str[::-1] == truth_str:
        return "Digit Reversal"
    
    # Check for missing digit
    if len(pred_str) == len(truth_str) - 1 and all(d in truth_str for d in pred_str):
        return "Missing Digit"
    
    # Check for extra digit
    if len(pred_str) == len(truth_str) + 1 and all(d in pred_str for d in truth_str):
        return "Extra Digit"
    
    return "Completely Wrong"

def calculate_cer(str1: str, str2: str) -> float:
    """Calculate Character Error Rate between two strings."""
    if not str1 or not str2:
        return 1.0  # Return maximum error if either string is empty
    
    # Convert to strings and remove whitespace
    str1 = str(str1).strip()
    str2 = str(str2).strip()
    
    # Calculate Levenshtein distance
    if len(str1) < len(str2):
        str1, str2 = str2, str1
    
    if len(str2) == 0:
        return 1.0
    
    previous_row = range(len(str2) + 1)
    for i, c1 in enumerate(str1):
        current_row = [i + 1]
        for j, c2 in enumerate(str2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    # Return CER as distance divided by length of longer string
    return previous_row[-1] / len(str1)

def analyze_results(results_file: str, ground_truth_file: str = "data/ground_truth.csv") -> dict:
    """Analyze model performance and generate analysis report."""
    import pandas as pd
    
    # Load results and ground truth
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Read ground truth with explicit string type for Invoice column
    ground_truth = pd.read_csv(ground_truth_file, dtype={'Invoice': str})
    
    # Debug ground truth data
    logger.info(f"Ground truth columns: {ground_truth.columns.tolist()}")
    logger.info(f"Ground truth shape: {ground_truth.shape}")
    logger.info(f"Sample of ground truth Invoice values: {ground_truth['Invoice'].head().tolist()}")
    
    # Initialize analysis structure
    analysis = {
        "metadata": results["metadata"],
        "summary": {
            "total_images": len(results["results"]),
            "completed": 0,
            "errors": 0,
            "work_order_accuracy": 0,
            "total_cost_accuracy": 0,
            "average_cer": 0
        },
        "error_categories": {
            "work_order": {},
            "total_cost": {}
        },
        "results": []
    }
    
    # Process each result
    total_cer = 0
    work_order_matches = 0
    total_cost_matches = 0
    
    for result in results["results"]:
        # Get ground truth for this image - remove .jpg extension for matching
        image_id = result["image_name"].replace(".jpg", "")
        
        # Debug image matching
        logger.info(f"Looking for image_id: {image_id}")
        logger.info(f"Available Invoice values: {ground_truth['Invoice'].tolist()}")
        
        gt_row = ground_truth[ground_truth["Invoice"] == image_id]
        
        if gt_row.empty:
            logger.warning(f"No ground truth found for image {image_id}")
            continue
            
        gt_work_order = str(gt_row["Work Order Number/Numero de Orden"].iloc[0]).strip()
        gt_total_cost = normalize_total_cost(str(gt_row["Total"].iloc[0]))
        
        # Initialize result analysis
        result_analysis = {
            "image_name": result["image_name"],
            "status": result["status"]
        }
        
        if result["status"] == "completed":
            analysis["summary"]["completed"] += 1
            
            # Analyze work order
            pred_work_order = result["extracted_data"]["work_order_number"]
            work_order_error = categorize_work_order_error(pred_work_order, gt_work_order)
            work_order_cer = calculate_cer(pred_work_order, gt_work_order)
            
            if work_order_error == "Exact Match":
                work_order_matches += 1
            
            # Analyze total cost
            pred_total_cost = normalize_total_cost(result["extracted_data"]["total_cost"])
            total_cost_error = categorize_total_cost_error(pred_total_cost, gt_total_cost)
            
            if total_cost_error == "Numeric Match":
                total_cost_matches += 1
            
            # Update result analysis
            result_analysis.update({
                "work_order": {
                    "predicted": pred_work_order,
                    "ground_truth": gt_work_order,
                    "error_category": work_order_error,
                    "cer": work_order_cer
                },
                "total_cost": {
                    "predicted": pred_total_cost,
                    "ground_truth": gt_total_cost,
                    "error_category": total_cost_error
                }
            })
            
            # Update error categories
            analysis["error_categories"]["work_order"][work_order_error] = \
                analysis["error_categories"]["work_order"].get(work_order_error, 0) + 1
            analysis["error_categories"]["total_cost"][total_cost_error] = \
                analysis["error_categories"]["total_cost"].get(total_cost_error, 0) + 1
            
            total_cer += work_order_cer
            
        else:
            analysis["summary"]["errors"] += 1
            result_analysis["error"] = result["error"]
        
        analysis["results"].append(result_analysis)
    
    # Calculate summary statistics
    total_images = analysis["summary"]["total_images"]
    if total_images > 0:
        analysis["summary"]["work_order_accuracy"] = work_order_matches / total_images
        analysis["summary"]["total_cost_accuracy"] = total_cost_matches / total_images
        analysis["summary"]["average_cer"] = total_cer / total_images
    
    return analysis

# %% [markdown]
"""
## Run Analysis
Generate and display analysis of model performance.
"""

# %%
def run_analysis():
    """Run the model and analyze its performance."""
    try:
        # Get test results file
        results_file = select_test_results_file()
        
        # Generate analysis
        analysis = analyze_results(str(results_file))
        
        # Create analysis directory if it doesn't exist
        analysis_dir = Path("analysis")
        analysis_dir.mkdir(exist_ok=True)
        
        # Save analysis to file
        analysis_file = analysis_dir / f"analysis_{analysis['metadata']['test_id']}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Display summary
        print("\nAnalysis Summary:")
        print("-" * 50)
        print(f"Total Images: {analysis['summary']['total_images']}")
        print(f"Completed: {analysis['summary']['completed']}")
        print(f"Errors: {analysis['summary']['errors']}")
        print(f"Work Order Accuracy: {analysis['summary']['work_order_accuracy']:.2%}")
        print(f"Total Cost Accuracy: {analysis['summary']['total_cost_accuracy']:.2%}")
        print(f"Average CER: {analysis['summary']['average_cer']:.3f}")
        
        print("\nWork Order Error Categories:")
        for category, count in analysis['error_categories']['work_order'].items():
            print(f"- {category}: {count}")
        
        print("\nTotal Cost Error Categories:")
        for category, count in analysis['error_categories']['total_cost'].items():
            print(f"- {category}: {count}")
        
        print(f"\nAnalysis saved to: {analysis_file}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

# Run the analysis
analysis_results = run_analysis()

