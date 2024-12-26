from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from datasets import load_dataset
import evaluate
from unsloth import FastLanguageModel
import wandb

# 1. Initialize wandb
wandb.init(
    project="qwen_cot_single_gpu_test",
    config={
        "learning_rate": 3e-5,
        "architecture": "Qwen2.5-7B",
        "dataset": "O1-OPEN/OpenO1-SFT-Ultra",
        "epochs": 1,
    }
)

# 2. Configuration section
max_seq_length = 1024   # Shorter for debugging
dtype = None            # Let model auto-select if needed (bf16 on H100 is typical)
load_in_4bit = False    # Could enable 4-bit quant to reduce memory usage

# 3. Load model & tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model.gradient_checkpointing_enable()

# 4. Load just a tiny slice of the dataset for testing
train_dataset = load_dataset("O1-OPEN/OpenO1-SFT-Ultra", split="train[:0.1%]")
eval_dataset = load_dataset("O1-OPEN/OpenO1-SFT-Ultra", split="train[0.1%:0.2%]")

print("Train Dataset columns:", train_dataset.column_names)
print("Eval Dataset columns:", eval_dataset.column_names)

# 5. Define the prompt template
instruction_template = """Below is a user query. 
Write a helpful, detailed response that completes the request.

### Query:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    # The new dataset columns are 'query' (like instruction) and 'response' (long chain-of-thought).
    # If you prefer *not* to include the full chain-of-thought, you can filter or mask it here.
    # For now, we'll fine-tune using the 'response' text.
    queries = examples["query"]
    responses = examples["response"]
    texts = []

    for q, r in zip(queries, responses):
        formatted_text = (
            instruction_template.format(q.strip(), r.strip()) + tokenizer.eos_token
        )
        texts.append(formatted_text)

    return {"text": texts}

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

# 6. Load metric
metric = evaluate.load("accuracy")  # Example metric. Replace if you'd like.

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 7. Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,   # Very small for testing
    gradient_accumulation_steps=4,   # Accumulate gradients
    warmup_steps=2,
    num_train_epochs=1,             # Just 1 epoch for a quick test
    learning_rate=3e-5,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),   # H100 typically supports bf16 well
    logging_steps=1,
    logging_dir="logs",
    optim="adamw_8bit",            # bitsandbytes required
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs_single_gpu",
    report_to="wandb",
    max_grad_norm=1.0,
    load_best_model_at_end=False,   # For a test run, we can skip best model logic
    eval_strategy="epoch",
    save_strategy="no",            # Don't save, since it's just a test
    run_name="qwen_cot_single_gpu_test_run"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=1,   # Fewer processes for quick test
    packing=False,        # Usually helpful, but can be off for testing
    args=training_args,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)

trainer.train()

wandb.finish()