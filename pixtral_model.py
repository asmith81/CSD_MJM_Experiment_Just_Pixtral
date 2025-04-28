# %% [markdown]
"""
# Pixtral Model Evaluation Notebook

This notebook evaluates the Pixtral-12B model's performance on invoice data extraction.
It follows the project's notebook handling rules and functional programming approach.
"""

# %% [markdown]
"""
## Setup and Configuration
"""

# %%
import os
import sys
import subprocess
from pathlib import Path
import logging
import json
import torch
from PIL import Image
from typing import Union, Dict, Any, List, Literal
import yaml

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

# %%
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %%
# Determine root directory
try:
    # When running as a script
    current_file = Path(__file__)
    ROOT_DIR = current_file.parent
    # Verify both required files exist
    if not (ROOT_DIR / "pixtral_model.py").exists() or not (ROOT_DIR / "requirements.txt").exists():
        raise RuntimeError("Could not find both pixtral_model.py and requirements.txt in the same directory")
except NameError:
    # When running in a notebook, look for the files in current directory
    current_dir = Path.cwd()
    if not (current_dir / "pixtral_model.py").exists() or not (current_dir / "requirements.txt").exists():
        raise RuntimeError("Could not find both pixtral_model.py and requirements.txt in the current directory")
    ROOT_DIR = current_dir

sys.path.append(str(ROOT_DIR))

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
        ("Base requirements", [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT_DIR / "requirements.txt")]),
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
    from transformers import AutoProcessor, LlavaForConditionalGeneration
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
                model_kwargs["load_in_8bit"] = True
            elif quantization == "int4":
                model_kwargs["load_in_4bit"] = True
            
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

# Download model
model, processor = download_pixtral_model()
logger.info("Model and processor ready for use")

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
    
    # Format the prompt with special tokens
    formatted_prompt = f"{special_tokens[2]}\n{prompt_text}\n{special_tokens[3]}"
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
    
    # Format the prompt
    prompt_text = SELECTED_PROMPT['prompts'][0]['text']
    formatted_prompt = format_prompt(prompt_text)
    
    # Display the image
    print("\nInput Image:")
    display(image)
    
    # Display the prompt
    print("\nFormatted Prompt:")
    print("-" * 50)
    print(formatted_prompt)
    print("-" * 50)
    
    # Prepare model inputs
    inputs = processor(
        text=formatted_prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)
    
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

