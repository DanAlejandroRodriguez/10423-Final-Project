# Test Time Scaling for Autonomous VLA Models

**10-423/623 Generative AI — Course Final Project**  
Cynthia Han · Daniel Rodriguez · Lucy Chiu  
Carnegie Mellon University

---

## Overview

Autonomous vehicles must make complex, safety-critical decisions in real time. While Vision-Language-Action (VLA) models equipped with Chain-of-Thought (CoT) reasoning have shown strong performance on driving benchmarks, their inference latency is often incompatible with the strict timing demands of real-world deployment.

This project investigates **test-time scaling strategies** for autonomous VLA inference by combining three complementary techniques:

| Technique | Purpose |
|---|---|
| **FastDriveCoT** parallel decoding | Baseline: parallelise independent CoT sub-tasks via a DAG scheduler |
| **Monte Carlo Tree Search (MCTS)** | Deeper trajectory exploration for complex scenes |
| **Difficulty-adaptive compute allocation** | Route simple scenes to fast parallel decoding; route complex scenes to MCTS |

As a stretch goal we plan to integrate **Block-Sparse FlashAttention (BSFA)** to reduce memory-bandwidth overhead for the dependency-graph attention masks.

---

## Baseline VLM — Gemma 4

Our baseline Vision-Language-Action model is **[Gemma 4](https://ai.google.dev/gemma)** (Google DeepMind).  
Gemma 4 is a multimodal open-weights model that accepts both image and text inputs, making it well-suited for driving perception and reasoning tasks.  
We connect Gemma 4 to the DriveLM task by:

1. **Prompt engineering** — wrapping each DriveLM QA pair in a structured reasoning template that mirrors FastDriveCoT's sub-task decomposition.
2. **DAG scheduler** — identifying independent perception sub-tasks (object detection, traffic-signal understanding, pedestrian intent, etc.) and dispatching them as parallel forward passes.
3. **MCTS integration** — treating each candidate trajectory token sequence as a node in the search tree, with the model's own confidence scores used as rollout rewards.

> **Why Gemma 4?** Gemma 4 provides a publicly available, permissively licensed multimodal backbone that is small enough to iterate on quickly while still being representative of production-scale VLMs.

---

## Dataset

We use **DriveLM** ([GitHub](https://github.com/OpenDriveLab/DriveLM)), derived from the **nuScenes** autonomous-driving dataset.

| Property | Detail |
|---|---|
| Sensor modalities | 6 cameras (surround view) |
| Scene conditions | Diverse road types, weather, and lighting |
| Annotations | Structured QA pairs with human CoT reasoning and ground-truth trajectories |
| Task format | Open-loop evaluation — model observes a keyframe sequence and predicts the trajectory |

### Evaluation Metrics

| Metric | Description |
|---|---|
| **CoT generation latency** | Wall-clock time to produce a full reasoning chain |
| **Meta-action IOU** | Intersection-over-union of predicted vs. ground-truth high-level actions |
| **Trajectory ADE** | Average Displacement Error over a 6.4-second prediction horizon |

---

## Project Structure

```
10423-Final-Project/
├── README.md               ← this file
├── requirements.txt        ← Python dependencies
├── data/                   ← DriveLM data pipeline
│   ├── __init__.py
│   ├── drivelm_dataset.py  ← Dataset loader (DriveLM / nuScenes)
│   ├── preprocess.py       ← Feature extraction & tokenisation helpers
│   └── README.md           ← Data setup instructions
├── models/                 ← Model components (to be implemented)
│   └── __init__.py
├── search/                 ← MCTS and adaptive-compute routing (to be implemented)
│   └── __init__.py
└── evaluation/             ← Evaluation scripts (to be implemented)
    └── __init__.py
```

---

## Approach

### 1 — Replicate FastDriveCoT Baseline
Reproduce the FastDriveCoT reasoning chain (perception → scene analysis → planning) using **Gemma 4** as the backbone, targeting the reported **3.1–4.1× CoT generation speedup** on DriveLM.

### 2 — MCTS-Guided Trajectory Search
Wrap the VLA forward pass inside an MCTS loop:
- **State** — partial CoT prefix + current driving context  
- **Action** — next reasoning token or trajectory waypoint  
- **Reward** — trajectory ADE on a held-out validation split (proxy: model log-probability)  
- **Policy/value network** — Gemma 4 itself (no separate critic during initial experiments)

### 3 — Difficulty-Adaptive Compute Allocation
Estimate scene difficulty from FastDriveCoT's enumeration stage (count of critical objects in view).  
Route:
- **Simple scenes** → single-pass parallel decoding (low latency)  
- **Complex scenes** → MCTS search (higher quality, bounded latency budget)

### Stretch Goal — Block-Sparse FlashAttention
Adapt the BSFA kernel to the DAG attention mask to further cut memory bandwidth during parallel sub-task decoding.

---

## Expected Outcomes

| Experiment | Hypothesis |
|---|---|
| FastDriveCoT replication | 3–4× CoT speedup vs. sequential decoding; ADE ≈ reported baseline |
| + MCTS | Improved ADE in complex multi-actor scenes; higher meta-action IOU |
| + Adaptive routing | Near-baseline latency on simple scenes; MCTS gains preserved for complex scenes |
| + BSFA (stretch) | Additional latency reduction on GPU |

---

## Team & Responsibilities

| Member | Phase 1 (Midway) | Phase 2 (Final) |
|---|---|---|
| **Daniel Rodriguez** | Custom causal attention masks & parallel forward pass | System integration, batching optimisations, BSFA (stretch) |
| **Cynthia Han** | DAG dependency-graph scheduler | MCTS algorithm & trajectory reward model |
| **Lucy Chiu** | DriveLM data pipeline & evaluation scripts | Difficulty-adaptive compute allocation |

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/DanAlejandroRodriguez/10423-Final-Project.git
cd 10423-Final-Project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download DriveLM data — see data/README.md for instructions
```

---

## References

- **FastDriveCoT** — parallel CoT decoding for autonomous driving VLAs  
- **Snell et al. (2024)** — Scaling LLM test-time compute optimally ([arXiv:2408.03314](https://arxiv.org/abs/2408.03314))  
- **MCTS for LLMs** — Monte Carlo Tree Search for decision-making in language models  
- **DriveLM** — Driving with Graph Visual Question Answering ([GitHub](https://github.com/OpenDriveLab/DriveLM))  
- **DriveVLM** — Tian et al., CoT-based hierarchical planning for autonomous driving  
- **BSFA** — Block-Sparse FlashAttention for efficient sparse attention  
- **Gemma 4** — Google DeepMind open-weights multimodal model  

---

## License

This project is developed for academic purposes as part of CMU 10-423/623 Generative AI (Spring 2025).
