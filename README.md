# Pixtral Model Evaluation

This project evaluates the Pixtral-12B model's performance on invoice data extraction. It follows a notebook-based approach with proper configuration and error handling.

## Features

- Configurable model quantization (bfloat16, int8, int4)
- Multiple prompt types for evaluation
- Memory resource monitoring
- Flash Attention support
- Device mapping configuration
- Single and multi-GPU support

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the model:
   - Select quantization level (bfloat16, int8, int4)
   - Choose prompt type
   - Set up device mapping

## Usage

1. Run the notebook cells in sequence:
   - Setup and configuration
   - Model initialization with selected quantization
   - Prompt selection
   - Single image test

2. The model will process images and extract information based on the selected prompt.

## Configuration

- Model parameters are in `config/pixtral.yaml`
- Prompts are stored in `config/prompts/`
- Images should be placed in `data/images/`

## Notes

- The model handles tensor type conversions internally based on quantization settings
- Memory requirements vary based on quantization level
- Flash Attention is automatically configured if supported by the GPU

## Error Handling

- Memory resource checks before model initialization
- Graceful handling of GPU availability
- Proper cleanup on initialization failures

## License

[Your License Here] 