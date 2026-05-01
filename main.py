"""
main.py — DriveLM evaluation pipeline.

Usage:
    python main.py --model 0 --limit 51          # baseline
    python main.py --model 1 --limit 51          # fastdrive
    python main.py --model 2 --limit 10           # mcts
"""

import os
import argparse
from data.drivelm_dataset import DriveLMDataset
from data.preprocess import PromptFormatter
from models.fastdrive import FastDriveVLA
from models.hybrid import HybridVLA
from evaluation import DriveLMEvaluator
from evaluation.metrics import _print_summary
import dotenv
dotenv.load_dotenv()

MODEL_NAMES = {0: "Baseline (AR)", 1: "FastDriveCoT", 2: "MCTS", 3: "Hybrid"}
RESULT_KEYS = {0: "baseline_ar_", 1: "fastdrivecot_", 2: "mcts_k{k}_", 3: "hybrid_"}


def find_real_frames(dataset, nuscenes_root, limit=None):
    dataroot = nuscenes_root
    indices = []
    for i, entry in enumerate(dataset.scene_list):
        if limit and len(indices) >= limit:
            break
        try:
            sample = dataset.nusc.get('sample', entry["token"])
            sd = dataset.nusc.get('sample_data', sample['data']['CAM_FRONT'])
            if os.path.exists(os.path.join(dataroot, sd['filename'])):
                indices.append(i)
        except Exception:
            pass
    return indices


def run_model(model, sample, model_id, mcts_iterations):
    images, question = sample["images"], sample["question"]
    if model_id == 0:
        return model.generate_trajectory(images, question)
    prompt = PromptFormatter.format(question, images=images)
    if model_id == 1:
        return model.generate_trajectory_parallel(images=images, text_prompt=prompt)
    if model_id == 2:
        return model.mcts_fastdrive_generate(images=images, text_prompt=prompt, iterations=mcts_iterations)
    num_objects = len(sample["answer"].get("key_object_infos", {}))
    return model.generate_trajectory_hybrid(
        images=images, text_prompt=prompt,
        num_critical_objects=num_objects, mcts_iterations=mcts_iterations,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, default=0, choices=[0, 1, 2, 3],
                        help="0=baseline  1=fastdrive  2=mcts  3=hybrid")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--nuscenes_root", type=str, default="/content/nuscenes/samples")
    parser.add_argument("--mcts_iterations", type=int, default=5)
    args = parser.parse_args()

    dataset = DriveLMDataset(split="train", nuscenes_img_dir=args.nuscenes_root)
    if args.model == 3:
        model = HybridVLA(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
    else:
        model = FastDriveVLA(model_id="Qwen/Qwen2.5-VL-7B-Instruct")

    real_indices = find_real_frames(dataset, args.nuscenes_root, args.limit)
    n = len(real_indices)
    print(f"Frames with real images: {n}")
    print(f"Running: {MODEL_NAMES[args.model]}\n")

    evaluator = DriveLMEvaluator()
    for count, i in enumerate(real_indices):
        sample = dataset[i]
        result = run_model(model, sample, args.model, args.mcts_iterations)

        print(f"[{count+1}/{n}] action={result.get('meta_action', '?'):14s}  "
              f"latency={result.get('latency_seconds', 0):.1f}s  "
              f"traj_pts={len(result.get('trajectory', []))}")

        evaluator.add(result, sample["gt_trajectory"], sample["answer"],
                      question=sample["question"], token=sample["token"])

    _print_summary(evaluator.summarise())
    key = RESULT_KEYS[args.model].format(k=args.mcts_iterations)
    evaluator.to_json(f"results_{key}.json")
    evaluator.to_json("results.json")


if __name__ == "__main__":
    main()
