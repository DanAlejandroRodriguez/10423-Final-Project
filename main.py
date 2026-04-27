"""
main.py — DriveLM evaluation pipeline.
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
from data.drivelm_dataset import DriveLMDataset
from data.preprocess import PromptFormatter
from models import QwenBaselineVLA, FastDriveVLA, HybridVLA
from evaluation import DriveLMEvaluator
from evaluation.metrics import _print_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--model", type=int, choices=[0, 1, 2], default=0,
                        help="0: Baseline, 1: MCTS Baseline, 2: FastDriveCoT")
    args = parser.parse_args()

    # Load data and model
    dataset = DriveLMDataset(split="train", nuscenes_img_dir="data/raw/nuscenes")
    
    if args.model == 0 or args.model == 1:
        print("Loading QwenBaselineVLA...")
        model = QwenBaselineVLA(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    if args.model == 2:
        print("Loading FastDriveVLA...")
        model = FastDriveVLA(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    elif args.model == 3:
        print("Loading HybridVLA...")
        model = HybridVLA(model_id="Qwen/Qwen2.5-VL-3B-Instruct")

    # Evaluate
    evaluator = DriveLMEvaluator()
    
    n_samples = len(dataset)
    if args.limit is not None:
        n_samples = min(n_samples, args.limit)
        print(f"Limiting evaluation to {n_samples} samples.")

    for i in range(n_samples):
        print(f"\n--- Evaluating sample {i+1}/{n_samples} ---")
        sample = dataset[i]
        
        # Pre-format the prompt (shared interface for all models)
        messages = PromptFormatter.format(question=sample["question"], images=sample["images"])
        
        if args.model == 0:
            result = model.generate_trajectory(sample["images"], messages)
        elif args.model == 1:
            result = model.mcts_generate(sample["images"], messages)
        elif args.model == 2:
            result = model.generate_trajectory_parallel(sample["images"], messages)
        elif args.model == 3:
            result = model.generate_trajectory_hybrid(sample["images"], messages)
        else:
            raise ValueError("Invalid model choice.")
        
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
