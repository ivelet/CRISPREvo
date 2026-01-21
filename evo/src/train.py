import os
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import load_from_disk
import wandb
from peft import LoraConfig, TaskType

from eval_metrics import compute_metrics
from model import EvoMultiClassTokenClassificationModel
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    # Retrieve the SLURM job ID from the environment variable
    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)

    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = os.path.join(ROOT_DIR, "crispr-datasets/evo")

    # Define hyperparameters
    hyperparameters = {
        # Tuning parameters
        "data_dir": os.path.join(DATA_DIR, "multi_data_only_bona_fide"),
        "eval_steps": 1,
        "eval_strategy": "epoch",
        "do_eval": False,
        "dropout_prob": 0.1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0001,
        "logging_steps": 10,
        "lr_scheduler_type": "cosine",
        "max_seq_length": 8000,
        # "max_steps": 500,
        "num_train_epochs": 3,
        "model_name": "togethercomputer/evo-1-8k-base",
        "per_device_eval_batch_size": 2,
        "per_device_train_batch_size": 2,
        "save_steps": 1000, # Note: each checkpoint is about 12.5GB (disabled here since > max_steps)
        "save_strategy": "epoch",
        "warmup_steps": 0,
        "weight_decay": 0,

        # LoRA-specific parameters
        "lora_task_type": TaskType.CAUSAL_LM, # Since the task for the evo model is still a causal LM, and the head is not affected by LoRA, we can keep this as is
        "lora_r": 1,                # Rank of LoRA matrices
        "lora_alpha": 32,           # Scaling factor for LoRA updates
        "lora_target_modules": [
            "Wqkv",                 # Query-Key-Value projection in AttentionBlock
            # "out_proj",           # Output projection in AttentionBlock
            # "projections",        # Projections in ParallelGatedConvBlock
            # "out_filter_dense",   # Dense layer after Hyena filter
            "l1",                   # First linear layer in MLP
            "l2",                   # Second linear layer in MLP
            "l3",                   # Third linear layer in MLP
        ],
        "lora_dropout": 0.05,       # Dropout rate for LoRA layers
        "lora_bias": "none",        # Bias configuration for LoRA ("none", "all", or specific layers)

        # Reproducibility parameters
        "bf16": True,
        "seed": 42,
        "SLURM_JOB_ID": slurm_job_id
    }

    # Initialize a new wandb run
    # wandb.init(project="crispr-evo", config=hyperparameters)

    lora_config = LoraConfig(
        task_type=hyperparameters["lora_task_type"],
        r=hyperparameters["lora_r"],
        lora_alpha=hyperparameters["lora_alpha"],
        target_modules=hyperparameters["lora_target_modules"],
        lora_dropout=hyperparameters["lora_dropout"],
        bias=hyperparameters["lora_bias"]
    )
    model = EvoMultiClassTokenClassificationModel(
        lora_config,
        dropout_prob=hyperparameters["dropout_prob"]
    )
    model.backbone_model.print_trainable_parameters()

    # In order to use PEFT with StripedHyena, that doesn't support input_embeds, we set it to a model type that PEFT knows
    model.backbone_model.config.model_type = "mpt"

    model.train()

    # Load the data
    dataset = load_from_disk(hyperparameters["data_dir"])

    tokenizer = AutoTokenizer.from_pretrained(hyperparameters["model_name"], trust_remote_code=True)
    tokenizer.pad_token = 'X'  
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=f"{ROOT_DIR}/results",
        eval_strategy=hyperparameters["eval_strategy"],
        learning_rate=hyperparameters["learning_rate"],
        lr_scheduler_type=hyperparameters["lr_scheduler_type"],
        weight_decay=hyperparameters["weight_decay"],
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
        per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
        num_train_epochs=hyperparameters["num_train_epochs"],
        logging_steps=hyperparameters["logging_steps"],
        eval_steps=hyperparameters["eval_steps"],
        do_eval=hyperparameters["do_eval"],
        save_strategy=hyperparameters["save_strategy"],
        save_steps=hyperparameters["save_steps"],
        warmup_steps=hyperparameters["warmup_steps"],
        bf16=hyperparameters["bf16"],
        report_to="wandb",
        seed=hyperparameters["seed"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # model.gradient_checkpointing_enable()
    trainer.args.save_strategy = "no"
    trainer.train()

    # In order to save the model, we need to reset the model type to the original one
    model.backbone_model.config.model_type = "stripedhyena"
    
    # Save the model
    model.save_pretrained(f"{ROOT_DIR}/results/multi/crispr_evo")

    # Finish the wandb run
    wandb.finish()
