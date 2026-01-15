import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import load_from_disk
import wandb
from peft import LoraConfig, get_peft_model, TaskType

from eval_metrics import compute_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    # Retrieve the SLURM job ID from the environment variable
    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)

    # Define hyperparameters
    hyperparameters = {
        # Tuning parameters
        "eval_steps": 1, # 500 default
        "eval_strategy": "epoch",
        "do_eval": False,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-04,
        "logging_steps": 10,
        "lr_scheduler_type": "cosine",
        "max_seq_length": 8000,
        # "max_steps": 500, # 10000 default
        "num_train_epochs": 3,
        "model_name": "togethercomputer/evo-1-8k-base",
        "per_device_eval_batch_size": 1,
        "per_device_train_batch_size": 1,
        "save_steps": 1000, # Each checkpoint is about 12.5GB (disabled here since > max_steps)
        "save_strategy": "epoch",
        "warmup_steps": 0,
        "weight_decay": 0,

        # LoRA-specific parameters
        "lora_task_type": TaskType.CAUSAL_LM,
        "lora_r": 1,                    # Rank of LoRA matrices
        "lora_alpha": 32,               # Scaling factor for LoRA updates
        "lora_target_modules": [
            "Wqkv",               # Query-Key-Value projection in AttentionBlock
            # "out_proj",           # Output projection in AttentionBlock
            # "projections",        # Projections in ParallelGatedConvBlock
            # "out_filter_dense",   # Dense layer after Hyena filter
            "l1",                 # First linear layer in MLP
            "l2",                 # Second linear layer in MLP
            "l3"                  # Third linear layer in MLP
        ],
        "lora_dropout": 0.05,           # Dropout rate for LoRA layers
        "lora_bias": "none",             # Bias configuration for LoRA ("none", "all", or specific layers)

        # Reproducibility parameters
        "bf16": True,
        "seed": 42,
        "SLURM_JOB_ID": slurm_job_id
    }

    # Initialize a new wandb run
    wandb.init(project="thesis", config=hyperparameters)

    model_config = AutoConfig.from_pretrained(hyperparameters["model_name"], trust_remote_code=True, revision="1.1_fix")
    model_config.use_cache = False

    print("Load base model for training")
    model = AutoModelForCausalLM.from_pretrained(hyperparameters["model_name"], config=model_config, trust_remote_code=True, revision="1.1_fix")

    # TODO create better solution for this
    # In order to use PEFT with StripedHyena, that doesn't support input_embeds, we set it to a model type that PEFT knows, that also doesn't support it.
    model.config.model_type = "mpt"

    model.train()

    # Apply LoRA configuration using hyperparameters
    lora_config = LoraConfig(
        task_type=hyperparameters["lora_task_type"],
        r=hyperparameters["lora_r"],
        lora_alpha=hyperparameters["lora_alpha"],
        target_modules=hyperparameters["lora_target_modules"],
        lora_dropout=hyperparameters["lora_dropout"],
        bias=hyperparameters["lora_bias"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Optional: Print trainable parameters for verification

    # Load the data
    dataset = load_from_disk("/work/dlclarge2/koeksalr-crispr/crispr-datasets")

    tokenizer = AutoTokenizer.from_pretrained(hyperparameters["model_name"], trust_remote_code=True)
    tokenizer.pad_token = 'X'  
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir="/work/dlclarge2/koeksalr-crispr/results",
        eval_strategy=hyperparameters["eval_strategy"],
        learning_rate=hyperparameters["learning_rate"],
        lr_scheduler_type=hyperparameters["lr_scheduler_type"],
        weight_decay=hyperparameters["weight_decay"],
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
        per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
        # max_steps=hyperparameters["max_steps"],
        num_train_epochs=hyperparameters["num_train_epochs"],
        logging_steps=hyperparameters["logging_steps"],
        eval_steps=hyperparameters["eval_steps"],
        do_eval=hyperparameters["do_eval"],
        save_strategy=hyperparameters["save_strategy"],
        resume_from_checkpoint='/work/dlclarge2/koeksalr-crispr/results/checkpoints',
        save_steps=hyperparameters["save_steps"],
        warmup_steps=hyperparameters["warmup_steps"],
        bf16=hyperparameters["bf16"],
        #fp16=hyperparameters["fp16"],
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

    model.gradient_checkpointing_enable()
    # trainer.args.save_strategy = "no"
    trainer.train()

    # TODO create better solution for this
    # In order to save the model, we need to reset the model type to the original one
    model.config.model_type = "stripedhyena"
    
    # Save the model
    model.save_pretrained(f"/work/dlclarge2/koeksalr-crispr/results/model-{slurm_job_id}")

    # Finish the wandb run
    wandb.finish()
