# %% [markdown]
"""
# Llama Vision Model Evaluation Notebook

This notebook evaluates the Llama-3.2-11B-Vision model's performance on invoice data extraction.
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
    if not MODEL_CONFIG or "prompt" not in MODEL_CONFIG:
        raise ValueError("Model configuration not loaded or missing prompt configuration")
    
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
            prompt_data = yaml.safe_load(f)
            # Apply Llama Vision prompt format
            for prompt in prompt_data['prompts']:
                prompt['text'] = MODEL_CONFIG['prompt']['format'].format(
                    prompt_text=prompt['text']
                )
            loaded_prompts[name] = prompt_data
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
    if not (ROOT_DIR / "llama_model.py").exists() or not (ROOT_DIR / "requirements_llama.txt").exists():
        raise RuntimeError("Could not find both llama_model.py and requirements_llama.txt in the same directory")
except NameError:
    # When running in a notebook, look for the files in current directory
    current_dir = Path.cwd()
    if not (current_dir / "llama_model.py").exists() or not (current_dir / "requirements_llama.txt").exists():
        raise RuntimeError("Could not find both llama_model.py and requirements_llama.txt in the current directory")
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
def install_dependencies():
    """Install required dependencies with progress tracking."""
    steps = [
        ("tqdm", [sys.executable, "-m", "pip", "install", "-q", "tqdm"]),
        ("Base requirements", [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT_DIR / "requirements_llama.txt")]),
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

# Import tqdm after installation
from tqdm import tqdm

# %% [markdown]
"""
## Model Configuration
Load and validate the Llama Vision model configuration.
"""

# %%
def load_model_config() -> Dict[str, Any]:
    """
    Load and validate the Llama Vision model configuration from YAML file.
    
    Returns:
        Dict containing the model configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
        ValueError: If required configuration sections are missing
    """
    config_path = Path("config/llama_vision.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {e}")
    
    # Validate required sections
    required_sections = [
        "name", "repo_id", "model_type", "processor_type",
        "hardware", "loading", "quantization", "prompt",
        "image_preprocessing", "inference"
    ]
    
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")
    
    # Validate hardware requirements
    if not config["hardware"].get("gpu_required"):
        logger.warning("GPU is required for optimal performance")
    
    # Log configuration summary
    logger.info(f"Loaded configuration for {config['name']}")
    logger.info(f"Model repository: {config['repo_id']}")
    logger.info(f"Hardware requirements: {config['hardware']}")
    
    return config

# Load model configuration
try:
    MODEL_CONFIG = load_model_config()
    logger.info("Model configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model configuration: {e}")
    raise

# %% [markdown]
"""
## Authentication Configuration
Configure Hugging Face authentication for accessing the gated model.
"""

# %%
def get_hf_token() -> str:
    """
    Get Hugging Face authentication token from environment or user input.
    
    Returns:
        str: Valid Hugging Face token
        
    Raises:
        ValueError: If token is invalid or missing required permissions
    """
    # Try to get token from environment
    token = os.getenv("HF_TOKEN")
    
    if not token:
        logger.info("HF_TOKEN not found in environment variables")
        token = input("\nPlease enter your Hugging Face token: ").strip()
    
    if not token:
        raise ValueError("No Hugging Face token provided")
    
    # Validate token format
    if not token.startswith("hf_"):
        raise ValueError("Invalid token format. Token should start with 'hf_'")
    
    # Test token with a simple API call
    try:
        import requests
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            "https://huggingface.co/api/models/meta-llama/Llama-3.2-11B-Vision",
            headers=headers
        )
        
        if response.status_code == 401:
            raise ValueError("Invalid token: Authentication failed")
        elif response.status_code == 403:
            raise ValueError("Token does not have access to meta-llama/Llama-3.2-11B-Vision")
        elif response.status_code != 200:
            raise ValueError(f"Token validation failed with status code: {response.status_code}")
            
        logger.info("Successfully validated Hugging Face token")
        return token
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to validate token: {str(e)}")

def configure_hf_auth():
    """Configure Hugging Face authentication for the session."""
    try:
        # Get and validate token
        token = get_hf_token()
        
        # Set token in environment for this session
        os.environ["HF_TOKEN"] = token
        
        # Configure huggingface_hub
        from huggingface_hub import HfFolder
        HfFolder.save_token(token)
        
        logger.info("Hugging Face authentication configured successfully")
        
    except Exception as e:
        logger.error(f"Failed to configure Hugging Face authentication: {e}")
        raise

# Configure Hugging Face authentication
try:
    configure_hf_auth()
except Exception as e:
    logger.error(f"Authentication configuration failed: {e}")
    raise

# %% [markdown]
"""
## Model Loading Configuration
Configure model loading parameters and device mapping.
"""

# %%
def configure_device_mapping() -> dict:
    """
    Configure device mapping based on available hardware and model requirements.
    
    Returns:
        dict: Device mapping configuration
        
    Raises:
        RuntimeError: If hardware requirements are not met
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. This model requires a GPU to run.")
    
    # Get GPU properties
    gpu_props = torch.cuda.get_device_properties(0)
    compute_capability = float(f"{gpu_props.major}.{gpu_props.minor}")
    
    # Check compute capability
    min_compute = float(MODEL_CONFIG["hardware"]["minimum_compute_capability"])
    if compute_capability < min_compute:
        raise RuntimeError(
            f"GPU compute capability {compute_capability} is below minimum required {min_compute}"
        )
    
    # Get GPU memory in GB
    gpu_memory = gpu_props.total_memory / (1024**3)
    logger.info(f"Available GPU memory: {gpu_memory:.2f}GB")
    
    # Configure device mapping based on quantization
    if QUANTIZATION_CONFIG.get("device_map") == "auto":
        # Use automatic device mapping
        device_map = "auto"
        logger.info("Using automatic device mapping")
    else:
        # Use single GPU setup
        device_map = {"": 0}
        logger.info("Using single GPU setup")
    
    return device_map

def get_model_loading_params() -> Dict[str, Any]:
    """
    Get model loading parameters based on configuration and selected quantization.
    
    Returns:
        dict: Model loading parameters
    """
    # Start with base parameters from config
    params = {
        "trust_remote_code": True,
        "use_auth_token": True,  # Required for Llama
        "device_map": configure_device_mapping()
    }
    
    # Add quantization parameters
    if "torch_dtype" in QUANTIZATION_CONFIG:
        params["torch_dtype"] = getattr(torch, QUANTIZATION_CONFIG["torch_dtype"])
    
    # Add flash attention if configured
    if QUANTIZATION_CONFIG.get("use_flash_attention_2", False):
        params["use_flash_attention_2"] = True
        params["attn_implementation"] = "flash_attention_2"
    
    # Add 4-bit quantization if selected
    if QUANTIZATION_CONFIG.get("load_in_4bit", False):
        from transformers import BitsAndBytesConfig
        params["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    
    # Add 8-bit quantization if selected
    if QUANTIZATION_CONFIG.get("load_in_8bit", False):
        from transformers import BitsAndBytesConfig
        params["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="fp8"
        )
    
    logger.info("Model loading parameters configured successfully")
    return params

# Initialize model loading parameters
try:
    MODEL_LOADING_PARAMS = get_model_loading_params()
    logger.info("Model loading configuration completed")
except Exception as e:
    logger.error(f"Failed to configure model loading parameters: {e}")
    raise

# %% [markdown]
"""
## Model Settings
Configure model settings including quantization.
"""

# %%
def get_quantization_config(selected_option: str = None) -> Dict[str, Any]:
    """
    Get quantization configuration for the selected option.
    If no option is selected, prompts the user to choose one.
    
    Args:
        selected_option: Optional pre-selected quantization option
        
    Returns:
        Dict containing the quantization configuration
    """
    if not MODEL_CONFIG or "quantization" not in MODEL_CONFIG:
        raise ValueError("Model configuration not loaded or missing quantization options")
    
    quantization_options = MODEL_CONFIG["quantization"]["options"]
    default_option = MODEL_CONFIG["quantization"]["default"]
    
    if not selected_option:
        print("\nAvailable quantization options:")
        for i, (option, config) in enumerate(quantization_options.items(), 1):
            memory_req = "16GB" if option == "bfloat16" else "8GB" if option == "int8" else "4GB"
            print(f"{i}. {option.upper()} (Memory: {memory_req})")
        
        while True:
            try:
                choice = int(input(f"\nSelect quantization (1-3) [default: {default_option}]: ") or "1")
                if 1 <= choice <= len(quantization_options):
                    selected_option = list(quantization_options.keys())[choice - 1]
                    break
                else:
                    print(f"Invalid choice. Please select a number between 1 and {len(quantization_options)}.")
            except ValueError:
                print("Please enter a valid number.")
    
    if selected_option not in quantization_options:
        raise ValueError(f"Invalid quantization option: {selected_option}")
    
    logger.info(f"Using {selected_option} quantization")
    return quantization_options[selected_option]

# Initialize quantization configuration
QUANTIZATION_CONFIG = get_quantization_config()
logger.info("Quantization configuration loaded successfully")

# %% [markdown]
"""
## Prompt and Processing Configuration
Configure prompt formatting, image processing, and inference parameters.
"""

# %%
def get_prompt_formatting_params() -> Dict[str, Any]:
    """
    Get prompt formatting parameters from configuration.
    
    Returns:
        dict: Prompt formatting parameters
    """
    if not MODEL_CONFIG or "prompt" not in MODEL_CONFIG:
        raise ValueError("Model configuration not loaded or missing prompt configuration")
    
    prompt_config = MODEL_CONFIG["prompt"]
    
    params = {
        "format": prompt_config["format"],
        "image_placeholder": prompt_config["image_placeholder"],
        "system_prompt": prompt_config["system_prompt"],
        "response_format": prompt_config["response_format"],
        "field_mapping": prompt_config["field_mapping"]
    }
    
    logger.info("Prompt formatting parameters configured successfully")
    return params

def get_image_processing_params() -> Dict[str, Any]:
    """
    Get image processing parameters from configuration.
    
    Returns:
        dict: Image processing parameters
    """
    if not MODEL_CONFIG or "image_preprocessing" not in MODEL_CONFIG:
        raise ValueError("Model configuration not loaded or missing image preprocessing configuration")
    
    img_config = MODEL_CONFIG["image_preprocessing"]
    
    params = {
        "max_size": tuple(img_config["max_size"]),
        "convert_to_rgb": img_config["convert_to_rgb"],
        "normalize": img_config["normalize"],
        "resize_strategy": img_config["resize_strategy"]
    }
    
    logger.info("Image processing parameters configured successfully")
    return params

def get_inference_params() -> Dict[str, Any]:
    """
    Get inference parameters from configuration.
    
    Returns:
        dict: Inference parameters
    """
    if not MODEL_CONFIG or "inference" not in MODEL_CONFIG:
        raise ValueError("Model configuration not loaded or missing inference configuration")
    
    inference_config = MODEL_CONFIG["inference"]
    
    params = {
        "max_new_tokens": inference_config["max_new_tokens"],
        "do_sample": inference_config["do_sample"],
        "temperature": inference_config["temperature"],
        "top_k": inference_config["top_k"],
        "top_p": inference_config["top_p"],
        "batch_size": inference_config["batch_size"],
        "max_batch_memory_gb": inference_config["max_batch_memory_gb"]
    }
    
    logger.info("Inference parameters configured successfully")
    return params

def format_prompt(prompt_text: str) -> str:
    """
    Format the prompt using the Llama Vision template.
    
    Args:
        prompt_text: The base prompt text
        
    Returns:
        str: Formatted prompt with system message and response format
    """
    prompt_params = get_prompt_formatting_params()
    
    # Format the prompt with special tokens and image token
    formatted_prompt = prompt_params["format"].format(
        prompt_text=prompt_text
    )
    
    return formatted_prompt

def process_image(image: Image.Image) -> Image.Image:
    """
    Process the image according to Llama Vision specifications.
    
    Args:
        image: Input PIL Image
        
    Returns:
        Image.Image: Processed image
    """
    img_params = get_image_processing_params()
    
    # Convert to RGB if needed
    if img_params["convert_to_rgb"] and image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize if needed
    if image.size[0] > img_params["max_size"][0] or image.size[1] > img_params["max_size"][1]:
        if img_params["resize_strategy"] == "maintain_aspect_ratio":
            image.thumbnail(img_params["max_size"], Image.Resampling.LANCZOS)
        else:
            image = image.resize(img_params["max_size"], Image.Resampling.LANCZOS)
    
    # Normalize if configured
    if img_params["normalize"]:
        # Convert to tensor for normalization
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
        # Convert back to PIL
        image = T.ToPILImage()(image)
    
    return image

# Initialize processing parameters
try:
    PROMPT_PARAMS = get_prompt_formatting_params()
    IMAGE_PARAMS = get_image_processing_params()
    INFERENCE_PARAMS = get_inference_params()
    logger.info("Processing parameters configured successfully")
except Exception as e:
    logger.error(f"Failed to configure processing parameters: {e}")
    raise

# %% [markdown]
"""
## Testing and Analysis
Functions for testing the model and analyzing results.
"""

# %%
def run_single_image_test(image_path: str = None) -> Dict[str, Any]:
    """
    Run the model on a single image using the selected prompt.
    
    Args:
        image_path: Optional path to image. If None, uses first .jpg from data/images
        
    Returns:
        Dict containing test results
    """
    try:
        # Get image path if not provided
        if not image_path:
            image_dir = Path("data/images")
            image_files = list(image_dir.glob("*.jpg"))
            if not image_files:
                raise FileNotFoundError("No .jpg files found in data/images directory")
            image_path = str(image_files[0])
        
        # Load and process image
        image = Image.open(image_path)
        processed_image = process_image(image)
        
        # Create a display version of the image
        display_image = processed_image.copy()
        max_display_size = (800, 800)
        display_image.thumbnail(max_display_size, Image.Resampling.LANCZOS)
        
        # Format the prompt
        prompt_text = SELECTED_PROMPT['prompts'][0]['text']
        formatted_prompt = format_prompt(prompt_text)
        
        # Display the image and prompt
        print("\nInput Image (resized for display):")
        display(display_image)
        
        print("\nFormatted Prompt:")
        print("-" * 50)
        print(formatted_prompt)
        print("-" * 50)
        
        # Prepare model inputs
        inputs = processor(
            text=formatted_prompt,
            images=[processed_image],
            return_tensors="pt"
        )
        
        # Move inputs to the correct device and dtype
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **INFERENCE_PARAMS
            )
        
        # Decode and display response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print("\nModel Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        return {
            "image_path": image_path,
            "prompt": formatted_prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during single image test: {str(e)}")
        raise

def process_batch(image_dir: str = "data/images") -> List[Dict[str, Any]]:
    """
    Process all images in the directory and collect responses.
    
    Args:
        image_dir: Directory containing images to process
        
    Returns:
        List of dictionaries containing results for each image
    """
    results = []
    image_files = list(Path(image_dir).glob("*.jpg"))
    
    if not image_files:
        raise FileNotFoundError(f"No .jpg files found in {image_dir}")
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            result = run_single_image_test(str(image_path))
            results.append(result)
            
            # Save incremental results
            save_incremental_results(results)
            
        except Exception as e:
            logger.error(f"Error processing image {image_path.name}: {str(e)}")
            results.append({
                "image_path": str(image_path),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            # Save incremental results even on error
            save_incremental_results(results)
    
    return results

def save_incremental_results(results: List[Dict[str, Any]]):
    """Save results incrementally to a file."""
    results_file = results_dir / f"test_results_{generate_test_id()}.json"
    
    # Create results data structure
    data = {
        "metadata": {
            "test_id": generate_test_id(),
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "name": MODEL_CONFIG["name"],
                "version": "1.0",
                "model_id": MODEL_CONFIG["repo_id"],
                "quantization": QUANTIZATION_CONFIG,
                "parameters": MODEL_LOADING_PARAMS
            },
            "prompt_type": selected_prompt_type,
            "system_resources": check_memory_resources()
        },
        "prompt": {
            "raw_text": SELECTED_PROMPT['prompts'][0]['text'],
            "formatted": format_prompt(SELECTED_PROMPT['prompts'][0]['text'])
        },
        "results": results
    }
    
    # Save to file
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")

def analyze_results(results_file: str) -> Dict[str, Any]:
    """
    Analyze model performance and generate analysis report.
    
    Args:
        results_file: Path to results file
        
    Returns:
        Dict containing analysis results
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
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
        if "error" in result:
            analysis["summary"]["errors"] += 1
            continue
            
        analysis["summary"]["completed"] += 1
        
        # Extract JSON from response
        try:
            # Find JSON in response
            response = result["response"]
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                extracted_data = json.loads(json_str)
                
                # Analyze work order
                work_order_error = categorize_work_order_error(
                    extracted_data.get("work_order_number", ""),
                    result.get("ground_truth", {}).get("work_order_number", "")
                )
                
                # Analyze total cost
                total_cost_error = categorize_total_cost_error(
                    normalize_total_cost(extracted_data.get("total_cost", "")),
                    normalize_total_cost(result.get("ground_truth", {}).get("total_cost", ""))
                )
                
                # Update statistics
                if work_order_error == "Exact Match":
                    work_order_matches += 1
                if total_cost_error == "Numeric Match":
                    total_cost_matches += 1
                
                # Update error categories
                analysis["error_categories"]["work_order"][work_order_error] = \
                    analysis["error_categories"]["work_order"].get(work_order_error, 0) + 1
                analysis["error_categories"]["total_cost"][total_cost_error] = \
                    analysis["error_categories"]["total_cost"].get(total_cost_error, 0) + 1
                
                # Calculate CER
                cer = calculate_cer(
                    extracted_data.get("work_order_number", ""),
                    result.get("ground_truth", {}).get("work_order_number", "")
                )
                total_cer += cer
                
                # Add to results
                analysis["results"].append({
                    "image_path": result["image_path"],
                    "status": "completed",
                    "work_order": {
                        "predicted": extracted_data.get("work_order_number", ""),
                        "ground_truth": result.get("ground_truth", {}).get("work_order_number", ""),
                        "error_category": work_order_error,
                        "cer": cer
                    },
                    "total_cost": {
                        "predicted": extracted_data.get("total_cost", ""),
                        "ground_truth": result.get("ground_truth", {}).get("total_cost", ""),
                        "error_category": total_cost_error
                    }
                })
                
        except Exception as e:
            logger.error(f"Error analyzing result: {str(e)}")
            analysis["results"].append({
                "image_path": result["image_path"],
                "status": "error",
                "error": str(e)
            })
    
    # Calculate summary statistics
    total_images = analysis["summary"]["total_images"]
    if total_images > 0:
        analysis["summary"]["work_order_accuracy"] = work_order_matches / total_images
        analysis["summary"]["total_cost_accuracy"] = total_cost_matches / total_images
        analysis["summary"]["average_cer"] = total_cer / total_images
    
    return analysis

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
