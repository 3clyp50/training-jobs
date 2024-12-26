import os
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from datasets import load_dataset
import evaluate
from unsloth import FastLanguageModel
import wandb

# 1. Initialize wandb
wandb.init(
    project="qwen_cot_full",
    config={
        "learning_rate": 2e-5,
        "architecture": "Qwen2.5-7B",
        "dataset": "O1-OPEN/OpenO1-SFT-Ultra",
        "epochs": 3,
    }
)

# Optional: Make sure any environment variables are set:
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# etc.

# 2. Configuration section
max_seq_length = 2048     # Larger for full training
dtype = None              # Let model auto-select if needed (bf16 likely on H100)
load_in_4bit = False      # Enable if you want more memory headroom

# 3. Load model & tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model.gradient_checkpointing_enable()

# 4. Load the full dataset
train_dataset = load_dataset("O1-OPEN/OpenO1-SFT-Ultra", split="train[:90%]")
eval_dataset = load_dataset("O1-OPEN/OpenO1-SFT-Ultra", split="train[90%:]")

print("Train Dataset columns:", train_dataset.column_names)
print("Eval Dataset columns:", eval_dataset.column_names)

# 5. Prompt template
instruction_template = """Below is a user query. 
Write a helpful, detailed response that completes the request.

### Query:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
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
    remove_columns=train_dataset.column_names,
    num_proc=4,  # Parallel map for speed
)

eval_dataset = eval_dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=eval_dataset.column_names,
    num_proc=4,
)

# 6. Metrics
metric = evaluate.load("accuracy")  # Or whichever metric is relevant

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 7. Training arguments
training_args = TrainingArguments(
    # Typical per-GPU batch size. If you find memory issues, reduce it or turn on 4-bit.
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=4,  
    warmup_steps=500,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=50,
    logging_dir="logs",
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs_full",
    report_to="wandb",
    max_grad_norm=1.0,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_strategy="epoch",
    save_total_limit=2,
    run_name="qwen_cot_full_finetune"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,
    packing=True,  # This can help utilize sequence space better
    args=training_args,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

wandb.finish()