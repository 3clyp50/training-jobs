from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from datasets import load_dataset
import evaluate
from unsloth import FastLanguageModel
import wandb

# 1. Initialize wandb
wandb.init(
    project="qwen_cot_finetune",
    config={
        "learning_rate": 3e-5,
        "architecture": "Qwen2.5-7B",
        "dataset": "O1-OPEN/OpenO1-SFT",
        "epochs": 3,
    }
)

# 2. Configuration section
max_seq_length = 2048    # Consider reducing to 1024 if memory is still too high
dtype = None             # Will let model auto-select if needed
load_in_4bit = False     # Set to True if you want 4-bit quantization

# 3. Load model & tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 4. Load datasets
#    Splitting 80/20 for demonstration purposes, adjust as needed.
train_dataset = load_dataset("O1-OPEN/OpenO1-SFT", split="train[:80%]")
eval_dataset = load_dataset("O1-OPEN/OpenO1-SFT", split="train[80%:]")

print("Train Dataset columns:", train_dataset.column_names)
print("Eval Dataset columns:", eval_dataset.column_names)

# 5. Define the prompt template
instruction_template = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []

    for instruction, output in zip(instructions, outputs):
        formatted_text = (
            instruction_template.format(instruction.strip(), output.strip())
            + tokenizer.eos_token
        )
        texts.append(formatted_text)

    return {"text": texts}

# 6. Map the datasets to the required prompt format
train_dataset = train_dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=train_dataset.column_names
)

eval_dataset = eval_dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=eval_dataset.column_names
)

# 7. Load metric using the 'evaluate' library
metric = evaluate.load("accuracy")  # Replace with a relevant metric for your use case

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 8. Set up training arguments
#    Adjust batch size and gradient accumulation to stay within GPU memory
training_args = TrainingArguments(
    per_device_train_batch_size=2,     # Lower if OOM persists (e.g., 1)
    gradient_accumulation_steps=4,     # Increase if lowering batch size
    warmup_steps=6,
    num_train_epochs=3,
    learning_rate=3e-5,
    fp16=not is_bfloat16_supported(),  # Or explicitly set fp16=False, bf16=True if certain
    bf16=is_bfloat16_supported(),      # H100 typically supports bf16
    logging_steps=10,
    logging_dir="logs",
    optim="adamw_8bit",               # Requires bitsandbytes
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="wandb",
    max_grad_norm=1.0,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_strategy="epoch",
    save_total_limit=2,
    run_name="qwen_cot_finetune_enhanced"
)

# 9. Create the SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,      # Adjust for faster dataset processing if needed
    packing=True,            # Packing can help reduce wasted sequence space
    args=training_args,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# 10. Train!
trainer.train()

# 11. Close wandb run
wandb.finish()