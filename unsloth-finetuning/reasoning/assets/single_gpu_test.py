import os
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported, FastLanguageModel
from datasets import load_dataset, concatenate_datasets
import evaluate
import wandb

def main():
    # -----------------------------
    # 1. Initialize Weights & Biases
    # -----------------------------
    wandb.init(
        project="qwen_cot_single_gpu",
        config={
            "learning_rate": 3e-5,
            "architecture": "Qwen2.5-7B",
            "dataset": "OpenO1-SFT-Ultra",
            "epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "dtype": "bf16" if is_bfloat16_supported() else "fp16",
            "load_in_4bit": False,
        }
    )

    # -----------------------------
    # 2. Configuration Parameters
    # -----------------------------
    max_seq_length = 2048          # Maximum sequence length
    dtype = "bf16" if is_bfloat16_supported() else "fp16"  # Optimal precision for H100
    load_in_4bit = False            # Toggle 4-bit loading if necessary

    # -----------------------------
    # 3. Load Model & Tokenizer
    # -----------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # -----------------------------
    # 4. Load and Prepare Dataset
    # -----------------------------
    # Load the dataset. Assuming it's available locally or via a repository.
    # For large datasets, streaming can be used if supported.
    train_split = "train"
    eval_split = "validation"  # Ensure your dataset has a validation split

    # Load training and evaluation datasets
    train_dataset = load_dataset("openo1-sft-ultra-35m-data", split=train_split, streaming=False)
    eval_dataset = load_dataset("openo1-sft-ultra-35m-data", split=eval_split, streaming=False)

    print("Train Dataset columns:", train_dataset.column_names)
    print("Eval Dataset columns:", eval_dataset.column_names)

    # Define the prompt template
    instruction_template = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

    # Define a function to format prompts
    def formatting_prompts_func(examples):
        instructions = examples["query"]
        responses = examples["response"]
        formatted_texts = [
            instruction_template.format(instr.strip(), resp.strip()) + tokenizer.eos_token
            for instr, resp in zip(instructions, responses)
        ]
        return {"text": formatted_texts}

    # Apply the formatting function to the datasets
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        batched=True,
        batch_size=1000,           # Adjust batch size for mapping efficiency
        remove_columns=train_dataset.column_names
    )

    eval_dataset = eval_dataset.map(
        formatting_prompts_func,
        batched=True,
        batch_size=1000,
        remove_columns=eval_dataset.column_names
    )

    # -----------------------------
    # 5. Load Evaluation Metric
    # -----------------------------
    metric = evaluate.load("accuracy")  # Choose a metric relevant to your task

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # -----------------------------
    # 6. Define Training Arguments
    # -----------------------------
    training_args = TrainingArguments(
        per_device_train_batch_size=2,          # Adjust based on GPU memory
        gradient_accumulation_steps=4,          # To simulate a larger batch size
        warmup_steps=500,                       # Number of warmup steps
        num_train_epochs=3,                     # Total number of training epochs
        learning_rate=3e-5,                     # Learning rate
        fp16=False,                             # Disable fp16 if bf16 is used
        bf16=is_bfloat16_supported(),           # Use bf16 for H100
        logging_steps=50,                       # Log every 50 steps
        logging_dir="logs",                     # Directory for logs
        optim="adamw_8bit",                     # 8-bit Adam optimizer
        weight_decay=0.01,                      # Weight decay
        lr_scheduler_type="linear",             # Learning rate scheduler
        seed=3407,                              # Random seed for reproducibility
        output_dir="outputs_single_gpu",        # Directory to save outputs
        report_to="wandb",                      # Report to Weights & Biases
        max_grad_norm=1.0,                      # Max gradient norm for clipping
        load_best_model_at_end=True,            # Load the best model at the end
        eval_strategy="epoch",                  # Evaluate at each epoch
        metric_for_best_model="eval_loss",       # Metric to determine the best model
        greater_is_better=False,                # Whether the metric is to be maximized
        save_strategy="epoch",                  # Save checkpoint at each epoch
        save_total_limit=2,                     # Save only the last 2 checkpoints
        run_name="qwen_cot_single_gpu_finetune" # Name for the run in wandb
    )

    # -----------------------------
    # 7. Initialize the Trainer
    # -----------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",              # The field containing the formatted text
        max_seq_length=max_seq_length,
        dataset_num_proc=4,                     # Number of processes for data loading
        packing=True,                           # Enable packing to optimize sequence space
        args=training_args,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # -----------------------------
    # 8. Start Training
    # -----------------------------
    trainer.train()

    # -----------------------------
    # 9. Save the Final Model
    # -----------------------------
    trainer.save_model("final_model_single_gpu")

    # -----------------------------
    # 10. Close WandB Run
    # -----------------------------
    wandb.finish()

if __name__ == "__main__":
    main()
