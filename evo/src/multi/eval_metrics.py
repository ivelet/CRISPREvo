import sys
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score
import torch
from model import EvoMultiClassTokenClassificationModel
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

def compute_metrics(eval_pred):
    """
    Compute multi-class classification metrics for HuggingFace Trainer.
    
    Args:
        eval_pred (tuple): (logits, labels) where:
            - logits: Shape (batch_size, seq_len, num_classes)
            - labels: Shape (batch_size, seq_len)

    Returns:
        dict: Metrics including accuracy, MCC, macro-averaged scores, and ROC AUC
    """
    logits, labels = eval_pred
    
    # Create boolean mask for valid tokens
    mask = labels != -100
    
    # Apply mask to get valid tokens
    masked_logits = logits[mask]  # (num_valid_tokens, num_classes)
    masked_labels = labels[mask]  # (num_valid_tokens,)
    
    # Compute probabilities and predictions
    probabilities = torch.softmax(torch.from_numpy(masked_logits), dim=-1).numpy()
    predictions = np.argmax(probabilities, axis=-1)
    
    # Calculate metrics
    accuracy = accuracy_score(masked_labels, predictions)
    mcc = matthews_corrcoef(masked_labels, predictions)
    
    # Macro-averaged scores
    precision, recall, f1, _ = precision_recall_fscore_support(
        masked_labels, predictions, average="macro"
    )
    
    # Multi-class ROC AUC (One-vs-Rest approach)
    # roc_auc = roc_auc_score(
    #     masked_labels, 
    #     probabilities,
    #     multi_class="ovr",
    #     average="macro"
    # )

    return {
        "accuracy": accuracy,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # "roc_auc": roc_auc,
    }


if __name__ == "__main__":
    # If argument is provided, set take the model id from the argument
    # Otherwise, use the default model
    model_id = None
    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    # Load the saved model
    model_dir = "/work/dlclarge2/koeksalr-crispr/results/multi/model-26970507"
    if model_id is not None:
        model_dir += f"-{model_id}"

    model = EvoMultiClassTokenClassificationModel.from_pretrained(model_dir)

    data_dir = "/work/dlclarge2/koeksalr-crispr/datasets/evo/multi_data_only_bona_fide/test"
    print(f"Loading data from {data_dir}")

    # Import the dataset
    test_dataset = load_from_disk(data_dir)

    # Discard the sequence column
    test_dataset = test_dataset.remove_columns("sequence")

    # Load the tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/evo-1-8k-base", trust_remote_code=True)
    tokenizer.pad_token = "X"
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # required, but not used
        report_to="none",  # Disable reporting
        per_device_eval_batch_size=1
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        eval_dataset=test_dataset,
        args=training_args
    )

    # Print out which model is being evaluated
    print(f"Evaluating model: {model_dir}")

    # TODO create better solution for this
    # In order to use PEFT with StripedHyena, that doesn't support input_embeds, we set it to a model type that PEFT knows, that also doesn't support it.
    model.backbone_model.config.model_type = "mpt"

    # Evaluate the model on the test set
    metrics = trainer.evaluate()

    # Save results to a file
    output_path = "/work/dlclarge2/koeksalr-crispr/results/test_eval"
    if model_id:
        output_path += f"-{model_id}"
    output_path += ".txt"

    with open(output_path, "w") as f:
        # For each metric in the metrics dictionary, write the metric name and value to the file
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")

    print(f"Results saved to {output_path}")
