from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from datasets import load_dataset
import evaluate
from unsloth import FastLanguageModel
import wandb

# Initialize wandb with your project configuration
wandb.init(
    project="qwen_cot_finetune",
    config={
        "learning_rate": 3e-5,
        "architecture": "Qwen2.5-7B",
        "dataset": "O1-OPEN/OpenO1-SFT",
        "epochs": 3,
    }
)

# Set up the model and tokenizer
max_seq_length = 2048
dtype = None 
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Load datasets
train_dataset = load_dataset("O1-OPEN/OpenO1-SFT", split="train[:80%]")
eval_dataset = load_dataset("O1-OPEN/OpenO1-SFT", split="train[80%:]")
print("Dataset columns:", train_dataset.column_names)

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

# Process the datasets
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

# Load metric using 'evaluate'
metric = evaluate.load("accuracy")  # Replace "accuracy" with your desired metric

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=6,
    num_train_epochs=3,
    learning_rate=3e-5,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=10,
    logging_dir="logs",
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="wandb",
    max_grad_norm=1.0,
    # Required for EarlyStoppingCallback
    load_best_model_at_end=True,
    # Use eval_strategy instead of evaluation_strategy (to avoid deprecation warning)
    eval_strategy="epoch",
    # Required when using load_best_model_at_end
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # Saving settings
    save_strategy="epoch",
    save_total_limit=2,
    # Run name for wandb
    run_name="qwen_cot_finetune"
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,
    packing=True,
    args=training_args,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Start training
trainer.train()

# Close wandb run when training is complete
wandb.finish()