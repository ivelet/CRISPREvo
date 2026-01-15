from datasets import load_from_disk
from pathlib import Path

def export_splits_to_csv(dataset_dir: str, output_dir: str = "./csv_splits"):
    """
    Exports Hugging Face dataset splits to separate CSV files
    
    Args:
        dataset_dir: Path to Hugging Face dataset directory
        output_dir: Output directory for CSV files (default: './csv_splits')
    """
    # Load dataset from disk
    dataset = load_from_disk(dataset_dir)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Handle different dataset structures
    if isinstance(dataset, dict):  # DatasetDict
        for split_name, split_data in dataset.items():
            file_path = output_path / f"{split_name}.csv"
            split_data.to_csv(str(file_path), index=False)
    else:  # Single Dataset
        file_path = output_path / "dataset.csv"
        dataset.to_csv(str(file_path), index=False)

# Take in two arguments when running the script
if __name__ == "__main__":
    import sys
    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./csv_splits"
    
    export_splits_to_csv(dataset_dir, output_dir)

