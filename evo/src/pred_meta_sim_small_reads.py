from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from model import EvoMultiClassTokenClassificationModel
from transformers import AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

parser = ArgumentParser(description="Evaluate a multi-class token classification model.")
parser.add_argument("--model_name", type=str, default="crispr_evo",
                    help="Model ID of a 150 target length model to infer with.")
parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size for evaluation.")
parser.add_argument("--data_path", type=str,
                    default=f"{ROOT_DIR}/data/simulated_reads_meta.csv",
                    help="Path to the dataset file containing simulated reads.")
parser.add_argument("--num_samples", type=int, default=100000,
                    help="Number of samples to process (default: 100k). Set to -1 for all samples.")

args = parser.parse_args()

# Load dataset
sim_small_reads_ds = load_dataset('csv', streaming=True,
                                    data_files={'train': args.data_path}, split="train")

# SHUFFLE AND TAKE SUBSET
if args.num_samples > 0:
    sim_small_reads_ds = sim_small_reads_ds.shuffle(seed=42, buffer_size=10000)
    sim_small_reads_ds = sim_small_reads_ds.take(args.num_samples)
    total_samples = args.num_samples
    print(f"Processing {args.num_samples} random samples from dataset")
else:
    print("Processing entire dataset (this will take ~20 hours)")
    # Count total if processing all
    import subprocess
    result = subprocess.run(['wc', '-l', args.data_path], stdout=subprocess.PIPE, text=True)
    total_samples = int(result.stdout.strip().split()[0]) - 1

total_batches = (total_samples + args.batch_size - 1) // args.batch_size

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/evo-1-8k-base", trust_remote_code=True)
tokenizer.pad_token = "X"

def collate_fn(batch):
    sequences = [item["sequence"] for item in batch]
    ids = [item["ID"] for item in batch]
    tokenized = tokenizer(sequences, return_tensors="pt")
    tokenized["ID"] = ids
    tokenized["sequence"] = sequences
    return tokenized

data_loader = DataLoader(sim_small_reads_ds, batch_size=args.batch_size, collate_fn=collate_fn)

# Load model
model_dir = f"{ROOT_DIR}/models/{args.model_name}"
model = EvoMultiClassTokenClassificationModel.from_pretrained(model_dir)
model.eval()
model.to("cuda")
model.backbone_model.config.model_type = "mpt"

output_dir = f"{ROOT_DIR}/results/metagenomic_inference"
Path(output_dir).mkdir(parents=True, exist_ok=True)
output_path = f"{output_dir}/infer_meta_{args.num_samples}.txt"

# Predict and save
with open(output_path, "w") as file:
    with torch.no_grad():
        for batch in tqdm(data_loader, total=total_batches, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            preds = torch.argmax(torch.softmax(logits, dim=-1), axis=-1).cpu().numpy()

            for i in range(len(batch["ID"])):
                file.write(f"ID:\t\t\t\t{batch['ID'][i]}\n")
                file.write(f"Sequence:\t\t{batch['sequence'][i]}\n")
                file.write(f"Predictions:\t{''.join(map(str, preds[i]))}\n\n")

print(f"Results saved to {output_path}")