"""
tests/test_integration_colab.py
================================
End-to-end integration test for Colab with blob 1 extracted.

What this tests
---------------
1. DriveLMDataset loads and finds real images from blob 1.
2. PromptFormatter builds a valid message dict.
3. Baseline (autoregressive) runs a full forward pass and returns structured output.
4. FastDriveCoT (parallel DAG decoding) runs and returns structured output.
5. Latency is compared between both paths.

Configuration via environment variables:
    NUSCENES_ROOT  — path to extracted nuScenes samples/ dir
    MODEL_ID       — HuggingFace model ID
    NUM_FRAMES     — number of frames to evaluate (default 3)

Run in Colab:
    import os
    os.environ["NUSCENES_ROOT"] = "/content/nuscenes/samples"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    !python tests/test_integration_colab.py

Or with pytest:
    !python -m pytest tests/test_integration_colab.py -v -s
"""

import os
import sys
import gc
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUSCENES_ROOT = os.environ.get("NUSCENES_ROOT", "/content/nuscenes/samples")
MODEL_ID      = os.environ.get("MODEL_ID",      "Qwen/Qwen2.5-VL-7B-Instruct")
NUM_FRAMES    = int(os.environ.get("NUM_FRAMES", "3"))


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


def evaluate_model(name, model_fn, dataset, test_frames, prompt_fn):
    """
    Run inference on test_frames using model_fn(images, text_prompt).
    Returns list of result dicts with latency, action, cot.
    Swap model_fn to test any model — baseline or FastDriveCoT.
    """
    print(f"\nRunning {name} on {len(test_frames)} frame(s)...\n")
    from data.preprocess import PromptFormatter

    results = []
    for i, entry in enumerate(test_frames):
        print(f"  Frame {i+1}/{len(test_frames)}  token={entry['token']}")

        item = dataset[dataset.scene_list.index(entry)]
        images   = item["images"]
        question = item["question"]
        gt_traj  = item["gt_trajectory"]

        print(f"    Real images : {count_real_images(dataset, entry)}/6")
        print(f"    QA pairs    : {len(entry['qas'])}")
        print(f"    GT traj pts : {len(gt_traj)}")

        text_prompt = PromptFormatter.format(question, num_images=len(images))
        result = model_fn(images, text_prompt)

        assert isinstance(result["latency_seconds"], float), "latency must be float"
        assert isinstance(result["raw_text"], str) and result["raw_text"], "raw_text empty"

        print(f"    Latency     : {result['latency_seconds']:.2f}s")
        print(f"    Action      : {result['meta_action']}")
        print(f"    CoT         : {str(result['chain_of_thought'])[:100]}")

        if result["chain_of_thought"] is None:
            print("    ⚠  No <cot> tag — expected before fine-tuning.")
        if result["meta_action"] is None:
            print("    ⚠  No <action> tag.")

        results.append(result)
        torch.cuda.empty_cache()
        print()

    return results


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_integration_test():
    print("=" * 60)
    print("DriveLM Integration Test")
    print(f"Model : {MODEL_ID}")
    print("=" * 60)

    # 1. Dataset
    print(f"\n[1/4] Loading DriveLMDataset ({NUSCENES_ROOT})...")
    from data.drivelm_dataset import DriveLMDataset
    dataset = DriveLMDataset(split="train", nuscenes_img_dir=NUSCENES_ROOT)
    print(f"      {len(dataset)} frames loaded.")

    # 2. Find frames with real images
    print("\n[2/4] Scanning for frames with extracted images...")
    real_frames = find_frames_with_real_images(dataset)
    if not real_frames:
        print(f"\n  ✗ No images found under {NUSCENES_ROOT}")
        print("  Run: from colab_setup import setup_colab; setup_colab()")
        return
    print(f"      Found {len(real_frames)} frames.")
    test_frames = real_frames[:NUM_FRAMES]

    # 3. Load model — FastDriveVLA supports both baseline and parallel paths
    print(f"\n[3/4] Loading {MODEL_ID}...")
    from models.fastdrive import FastDriveVLA
    model = FastDriveVLA(model_id=MODEL_ID)

    # 4. Baseline path — swap model.generate_trajectory for any other model here
    baseline_results = evaluate_model(
        name="[4/5] Baseline (autoregressive)",
        model_fn=lambda images, prompt: model.generate_trajectory(
            images=images, text_prompt=prompt, max_new_tokens=512
        ),
        dataset=dataset,
        test_frames=test_frames,
        prompt_fn=None,
    )

    gc.collect()
    torch.cuda.empty_cache()

    # 5. FastDriveCoT parallel path — swap model.generate_trajectory_parallel here
    fastdrive_results = evaluate_model(
        name="[5/5] FastDriveCoT (parallel DAG)",
        model_fn=lambda images, prompt: model.generate_trajectory_parallel(
            images=images, text_prompt=prompt
        ),
        dataset=dataset,
        test_frames=test_frames,
        prompt_fn=None,
    )

    for r in fastdrive_results:
        assert r.get("model_type") == "Qwen2.5VL_FastDriveCoT_Parallel", "wrong model_type"

    # Summary
    avg_baseline  = sum(r["latency_seconds"] for r in baseline_results)  / len(baseline_results)
    avg_fastdrive = sum(r["latency_seconds"] for r in fastdrive_results) / len(fastdrive_results)
    speedup = avg_baseline / avg_fastdrive if avg_fastdrive > 0 else float('nan')

    print("=" * 60)
    print(f"  Avg baseline latency    : {avg_baseline:.2f}s")
    print(f"  Avg FastDriveCoT latency: {avg_fastdrive:.2f}s")
    print(f"  Speedup                 : {speedup:.2f}x  (paper target: 3-4x)")
    print("=" * 60)
    print("✓ Pipeline end-to-end verified.")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_test()
