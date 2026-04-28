"""
tests/test_integration_colab.py
================================
End-to-end integration test for Colab with blob 1 extracted.

Models
------
  1 — Baseline (autoregressive)
  2 — FastDriveCoT (parallel DAG decoding)
  3 — FastDriveCoT-MCTS (MCTS over DAG waves)

Usage
-----
Run a single model:
    !python tests/test_integration_colab.py --models 2

Run all three:
    !python tests/test_integration_colab.py --models 1 2 3

Or via pytest (runs all by default):
    !python -m pytest tests/test_integration_colab.py -v -s

Configuration via environment variables:
    NUSCENES_ROOT   — path to extracted nuScenes samples/ dir
    MODEL_ID        — HuggingFace model ID
    NUM_FRAMES      — number of frames to evaluate (default 3)
    MCTS_ITERATIONS — MCTS rollout budget (default 5)
"""

import os
import sys
import gc
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUSCENES_ROOT   = os.environ.get("NUSCENES_ROOT",   "/content/nuscenes/samples")
MODEL_ID        = os.environ.get("MODEL_ID",        "Qwen/Qwen2.5-VL-7B-Instruct")
NUM_FRAMES      = int(os.environ.get("NUM_FRAMES",      "3"))
MCTS_ITERATIONS = int(os.environ.get("MCTS_ITERATIONS", "5"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_frames_with_real_images(dataset, max_search=500):
    """Return frames whose CAM_FRONT image exists on disk (extracted blob)."""
    found = []
    dataroot = os.path.dirname(dataset.img_dir)
    for entry in dataset.scene_list[:max_search]:
        try:
            sample = dataset.nusc.get('sample', entry["token"])
            sd = dataset.nusc.get('sample_data', sample['data']['CAM_FRONT'])
            if os.path.exists(os.path.join(dataroot, sd['filename'])):
                found.append(entry)
        except Exception:
            pass
    return found


def count_real_images(dataset, entry):
    """Count how many of the 6 camera images exist on disk for this frame."""
    dataroot = os.path.dirname(dataset.img_dir)
    count = 0
    try:
        sample = dataset.nusc.get('sample', entry["token"])
        for cam in dataset.CAMERA_VIEWS:
            sd = dataset.nusc.get('sample_data', sample['data'][cam])
            if os.path.exists(os.path.join(dataroot, sd['filename'])):
                count += 1
    except Exception:
        pass
    return count


def evaluate_model(name, model_fn, dataset, test_frames):
    """
    Run inference on test_frames using model_fn(images, prompt).
    Returns list of result dicts.
    """
    print(f"\nRunning {name} on {len(test_frames)} frame(s)...\n")
    results = []
    for i, entry in enumerate(test_frames):
        print(f"  Frame {i+1}/{len(test_frames)}  token={entry['token']}")

        item     = dataset[dataset.scene_list.index(entry)]
        images   = item["images"]
        question = item["question"]
        gt_traj  = item["gt_trajectory"]

        print(f"    Real images : {count_real_images(dataset, entry)}/6")
        print(f"    QA pairs    : {len(entry['qas'])}")
        print(f"    GT traj pts : {len(gt_traj)}")

        result = model_fn(images, question)

        assert isinstance(result["latency_seconds"], float), "latency must be float"
        assert isinstance(result["raw_text"], str) and result["raw_text"], "raw_text empty"

        print(f"    Latency     : {result['latency_seconds']:.2f}s")
        print(f"    Action      : {result['meta_action']}")
        print(f"    CoT         : {str(result['chain_of_thought'])[:120]}")

        if result["chain_of_thought"] is None:
            print("    ⚠  No <cot> tag — expected before fine-tuning.")
        if result["meta_action"] is None:
            print("    ⚠  No <action> tag.")

        results.append(result)
        torch.cuda.empty_cache()
        print()

    return results


def print_summary(label_results_pairs):
    """Print a latency + action IOU summary table for the given (label, results) pairs."""
    from evaluation.evaluate import compute_action_iou

    def gt_actions(entry):
        return [
            qa["answer"] for qa in entry["qas"]
            if any(kw in qa.get("question", "").lower()
                   for kw in ("action", "safe", "target"))
        ]

    print("\n" + "=" * 60)
    print(f"  {'Model':<26}  {'Action IOU':>10}  {'Avg Latency':>12}")
    print("-" * 60)
    for label, results, frames in label_results_pairs:
        ious = [compute_action_iou(r["meta_action"], gt_actions(e))
                for r, e in zip(results, frames)]
        lats = [r["latency_seconds"] for r in results]
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        mean_lat = sum(lats) / len(lats) if lats else 0.0
        print(f"  {label:<26}  {mean_iou:>10.3f}  {mean_lat:>11.2f}s")
    print("=" * 60)

    if len(label_results_pairs) > 1:
        base_lat = sum(r["latency_seconds"] for r in label_results_pairs[0][1]) / len(label_results_pairs[0][1])
        print()
        for label, results, _ in label_results_pairs[1:]:
            avg = sum(r["latency_seconds"] for r in results) / len(results)
            speedup = base_lat / avg if avg > 0 else float("nan")
            print(f"  {label} speedup vs {label_results_pairs[0][0]}: {speedup:.2f}x")

    if len(label_results_pairs) >= 2:
        print()
        header = f"    {'Frame':<6}" + "".join(f"  {lbl[:14]:<14}" for lbl, _, _ in label_results_pairs)
        print(header)
        n_frames = len(label_results_pairs[0][1])
        for i in range(n_frames):
            row = f"    {i+1:<6}"
            for _, results, _ in label_results_pairs:
                row += f"  {str(results[i]['meta_action']):<14}"
            print(row)

    print("=" * 60)


# ---------------------------------------------------------------------------
# Model runners  (one function per model so they can be called independently)
# ---------------------------------------------------------------------------

def run_baseline(model, dataset, test_frames):
    return evaluate_model(
        name="Baseline (autoregressive)",
        model_fn=lambda images, prompt: model.generate_trajectory(
            images=images, question=prompt, max_new_tokens=512
        ),
        dataset=dataset,
        test_frames=test_frames,
    )


def run_fastdrive(model, dataset, test_frames):
    from data.preprocess import PromptFormatter
    return evaluate_model(
        name="FastDriveCoT (parallel DAG)",
        model_fn=lambda images, prompt: model.generate_trajectory_parallel(
            images=images,
            text_prompt=PromptFormatter.format(prompt, images=images),
        ),
        dataset=dataset,
        test_frames=test_frames,
    )


def run_mcts(model, dataset, test_frames, iterations=MCTS_ITERATIONS):
    from data.preprocess import PromptFormatter
    print(f"\n  MCTS_ITERATIONS={iterations} — set env var to increase for better quality")
    return evaluate_model(
        name=f"FastDriveCoT-MCTS (k={iterations})",
        model_fn=lambda images, prompt: model.mcts_fastdrive_generate(
            images=images,
            text_prompt=PromptFormatter.format(prompt, images=images),
            iterations=iterations,
        ),
        dataset=dataset,
        test_frames=test_frames,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_integration_test(models_to_run=(1, 2, 3)):
    models_to_run = set(models_to_run)

    print("=" * 60)
    print("DriveLM Integration Test")
    print(f"Model  : {MODEL_ID}")
    print(f"Running: {sorted(models_to_run)}  (1=baseline, 2=fastdrive, 3=mcts)")
    print("=" * 60)

    # 1. Dataset
    print(f"\n[1] Loading DriveLMDataset ({NUSCENES_ROOT})...")
    from data.drivelm_dataset import DriveLMDataset
    dataset = DriveLMDataset(split="train", nuscenes_img_dir=NUSCENES_ROOT)
    print(f"    {len(dataset)} frames loaded.")

    # 2. Find frames with real images
    print("\n[2] Scanning for frames with extracted images...")
    real_frames = find_frames_with_real_images(dataset)
    if not real_frames:
        print(f"\n  ✗ No images found under {NUSCENES_ROOT}")
        print("  Run: from colab_setup import setup_colab; setup_colab()")
        return
    print(f"    Found {len(real_frames)} frames.")
    test_frames = real_frames[:NUM_FRAMES]

    # 3. Load model
    print(f"\n[3] Loading {MODEL_ID}...")
    from models.fastdrive import FastDriveVLA
    model = FastDriveVLA(model_id=MODEL_ID)

    # 4–6. Run selected models
    label_results = []   # list of (label, results, frames) for summary table

    if 1 in models_to_run:
        results = run_baseline(model, dataset, test_frames)
        for r in results:
            assert r.get("model_type") == "Qwen2.5VL_Autoregressive_Baseline", "wrong model_type"
        label_results.append(("Baseline (AR)", results, test_frames))
        gc.collect()
        torch.cuda.empty_cache()

    if 2 in models_to_run:
        results = run_fastdrive(model, dataset, test_frames)
        for r in results:
            assert r.get("model_type") == "Qwen2.5VL_FastDriveCoT_Parallel", "wrong model_type"
        label_results.append(("FastDriveCoT", results, test_frames))
        gc.collect()
        torch.cuda.empty_cache()

    if 3 in models_to_run:
        results = run_mcts(model, dataset, test_frames)
        for r in results:
            assert r.get("model_type") == "Qwen2.5VL_FastDriveCoT_MCTS", "wrong model_type"
        label_results.append((f"MCTS (k={MCTS_ITERATIONS})", results, test_frames))
        gc.collect()
        torch.cuda.empty_cache()

    if label_results:
        print_summary(label_results)

    print("\n✓ Done.")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="DriveLM integration test")
    parser.add_argument(
        "--models", nargs="+", type=int, default=[1, 2, 3],
        metavar="N",
        help="Which models to run: 1=baseline  2=fastdrive  3=mcts  (default: 1 2 3)",
    )
    return parser.parse_args()


# pytest entry point — runs all three by default
def test_integration():
    run_integration_test(models_to_run=(1, 2, 3))


if __name__ == "__main__":
    args = _parse_args()
    run_integration_test(models_to_run=args.models)
