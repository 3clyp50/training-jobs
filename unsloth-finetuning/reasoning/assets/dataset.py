from datasets import load_dataset
from unsloth import FastLanguageModel
import torch

# First set up the model and tokenizer
max_seq_length = 2048
dtype = None 
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Load dataset
dataset = load_dataset("O1-OPEN/OpenO1-SFT", split="train")
print("Dataset columns:", dataset.column_names)

# Define the prompt template
instruction_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    
    for instruction, output in zip(instructions, outputs):
        formatted_text = instruction_template.format(
            instruction.strip(),
            output.strip()
        ) + tokenizer.eos_token
        texts.append(formatted_text)
    
    return {"text": texts}

# Process the dataset
dataset = dataset.map(
    formatting_prompts_func, 
    batched=True,
    remove_columns=dataset.column_names
)
