"""
main.py — DriveLM evaluation pipeline.
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
from data.drivelm_dataset import DriveLMDataset
from models import QwenBaselineVLA, FastDriveVLA
from evaluation import DriveLMEvaluator
from evaluation.metrics import _print_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--model", type=int, choices=[0, 1], default=0, help="0: QwenBaselineVLA, 1: FastDriveVLA")
    args = parser.parse_args()

    # Load data and model
    dataset = DriveLMDataset(split="train", nuscenes_img_dir="data/raw/nuscenes")
    
    if args.model == 1:
        print("Loading FastDriveVLA...")
        model = FastDriveVLA(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    else:
        print("Loading QwenBaselineVLA...")
        model = QwenBaselineVLA(model_id="Qwen/Qwen2.5-VL-3B-Instruct")

    # Evaluate
    evaluator = DriveLMEvaluator()
    
    n_samples = len(dataset)
    if args.limit is not None:
        n_samples = min(n_samples, args.limit)
        print(f"Limiting evaluation to {n_samples} samples.")

    for i in range(n_samples):
        print(f"\n--- Evaluating sample {i+1}/{n_samples} ---")
        sample = dataset[i]
        if args.model == 1:
            result = model.generate_trajectory_parallel(sample["images"], sample["question"])
        else:
            result = model.generate_trajectory(sample["images"], sample["question"])
        
        print("\n--- MODEL GENERATION ---")
        print(result.get("raw_text", ""))
        print("------------------------\n")
        
        evaluator.add(result, sample["gt_trajectory"], sample["answer"],
                      question=sample["question"], token=sample["token"])

    # Print results and save
    _print_summary(evaluator.summarise())
    evaluator.to_json("results.json")


if __name__ == "__main__":
    main()
