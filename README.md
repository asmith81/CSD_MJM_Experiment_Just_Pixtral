# Pixtral Model Evaluation

This notebook evaluates the Pixtral-12B model's performance on invoice data extraction. It follows a structured approach to model configuration and setup.

## Environment Setup

The notebook handles the following setup steps:

1. **Dependencies Installation**
   - Installs required packages from requirements.txt
   - Configures PyTorch with CUDA 11.8 support
   - Sets up logging configuration

2. **Memory Resource Check**
   - Verifies GPU availability
   - Checks available GPU memory
   - Checks system RAM
   - Provides recommendations based on available resources

3. **Quantization Selection**
   - Offers three quantization options:
     - bfloat16 (full precision, 93GB VRAM)
     - int8 (8-bit, 46GB VRAM)
     - int4 (4-bit, 23GB VRAM)
   - Allows user to select appropriate quantization level

4. **Flash Attention Configuration**
   - Checks GPU support for Flash Attention
   - Automatically enables Flash Attention if supported
   - Provides feedback on Flash Attention status

5. **Device Mapping**
   - Configures single GPU setup by default
   - Optional multi-GPU support for parallel processing
   - Provides clear error if no GPU is available

6. **Version Checks**
   - Verifies all required package versions
   - Enforces transformers >=4.45.0 requirement
   - Checks critical dependencies:
     - transformers
     - Pillow
     - torch
     - accelerate
     - bitsandbytes
     - flash-attn
   - Provides warnings for version mismatches

7. **Model Download**
   - Downloads Pixtral-12B model from HuggingFace
   - Implements retry logic (2 attempts)
   - Monitors memory usage during download
   - Applies selected quantization settings
   - Returns both model and processor
   - Handles critical error cases

## Requirements

- Python 3.11
- CUDA 11.8
- NVIDIA GPU with sufficient VRAM (minimum 23GB for 4-bit quantization)
- See requirements.txt for package dependencies

## Usage

1. Run the notebook cells in sequence
2. Select quantization level when prompted
3. The notebook will automatically:
   - Check package versions
   - Configure memory resources
   - Enable Flash Attention (if supported)
   - Set up device mapping
   - Download and configure the model
4. For multi-GPU systems, uncomment the multi-GPU configuration cell

## Notes

- The notebook requires a GPU to run
- Memory requirements vary based on quantization level
- Flash Attention is enabled by default if supported
- Multi-GPU support is optional and commented out by default
- Model download includes retry logic and memory monitoring
- Package version checks ensure compatibility 