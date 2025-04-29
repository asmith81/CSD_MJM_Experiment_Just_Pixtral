 "Changing root finding logic": This actually works through model download
 "Added the prompt choice cell and single test inference cell.": 
 "Added headers in beginning": Made cosmetic changes but prompt cell didn't do anything (because the function was never run)
 "Run the select_prompt() function": turns out the function was run later in the notebook
 "Haha, the llm was smarter": Restores the prompt function to running later in the notebook
 "Instantiated Model, Removed Input Type Conversion": So in trouble shooting we ran into a type error for the model input tensors.  Tracing it back I realized we never properly instantiated the model before trying to run the single-image test.  To correct this we wrote a cell to instantiate the model with quantization level set by the global quantization variable defined earlier.  Then we checked the model card and it appears that the type conversion is handled by the processor.  So we removed the type conversion in the function for testing the model.  
  "Moving on to TS the single test": Just updating the project after corruption found on the hard drive with the git repo.  Corrected the corruption on the hard drive.  Accepted changes to the readme.  No moving on to test
  