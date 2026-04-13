# Test Time Scaling for Autonomous VLA Models

**Cynthia Han (csh2@andrew.cmu.edu) & Daniel Rodriguez (darodri2@andrew.cmu.edu) & Lucy Chiu (lchiu@andrew.cmu.edu)**  
10-423/623 Generative AI Course Project

---

## Abstract

Autonomous vehicles are a rapidly growing field, with major robotaxi companies deploying driverless rideshare services such as Waymo and Zoox, alongside automakers integrating advanced driver-assistance features into consumer vehicles. However, these systems must operate under strict real-world constraints, particularly in terms of real-time decision-making speed. Current CoT generation introduces inference latency that is incompatible. This proposal outlines our approach to addressing this bottleneck, by integrating FastDriveCoT parallel decoding with Monte Carlo Tree Search (MCTS) and difficulty adaptive compute allocation. This proposal aims to demonstrate significant inference speedups, high trajectory quality, and timely completion of our project.

---

## Introduction

Our project aims to investigate a variety of existing techniques that are known to optimize test-time performance and apply them to the application of autonomous VLA models. While parallel CoT decoding is native to autonomous driving, we are adapting difficulty adaptive compute allocation and Monte Carlo Tree Search (MCTS), techniques originally designed for general LLM reasoning, into domain. In this proposal, we outline the necessary steps to complete this project. We first discuss the DriveLM dataset that will be utilized and the tasks we hope to replicate from past studies. We then describe our approach to integrating these methods and the expected results following the completion of our experiments. We also outline how the tasks for this project will be distributed amongst all team members leading up to the Midway Executive Summary deadline.

---

## Dataset / Task

We aim to use the DriveLM dataset, derived from nuScenes, which features multi-sensor camera systems with different road and weather conditions and structured reasoning annotations. The dataset is comprised of keyframes capturing different driving scenarios, with different pedestrian and traffic patterns. The dataset is organized into short driving segments, with individual segments paired with Question-and-Answer logic suited for training perception and Chain-of-Thought. On the other hand, DriveLM provides ground-truth human reasoning and motion paths. This allows us to do open-loop evaluation where the model can make observations, generate candidate trajectories, and be compared to the human driver's choices.

The FastDriveCoT paper trains a VLA model utilizing a base Qwen LLM model. Our current plan is to utilize the DriveLM dataset to be able to train the VLA model to observe surroundings on real-world driving logs, and validate the MCTS-guided policies by measuring CoT generation latency, meta-action Intersection Over Union (IOU), and trajectory Average Displacement Error (ADE) over a 6.4-second horizon.

Our project relies on pre-existing open-source foundations. We will utilize the **Gemma 4** model as our baseline VLA architecture. Second, as a stretch goal, we intend to integrate Block-Sparse FlashAttention (BSFA) to further optimize memory bandwidth as this was a limitation that FastDriveCoT cites.

---

## Related Work

There are a wide variety of research papers that have focused on improving autonomous vehicle models. Tian et al. introduced DriveVLM, which proposed CoT-based scene description, scene analysis, and hierarchical planning. Due to the high inference latency of VLMs, they used a dual-system architecture to manage real-time control.

One specific technique has been developed by FastDriveCoT, which accelerated chain-of-thought (CoT) reasoning in autonomous vehicles. This reasoning method parallelized independent tasks that the VLA model recognized, including recognizing pedestrians and understanding traffic signals. Using a dependency graph to organize subtasks and decrease the amount of sequential processing, FastDriveCoT was able to obtain a 3–4x speedup in CoT generation; this implies an improvement in speed while maintaining performance.

UniAD (Hu et al., 2023) introduced a unified autonomous driving framework that jointly models perception, prediction, and planning end-to-end using transformers. By coupling bird's-eye-view features across tasks, UniAD demonstrated that tightly integrated representations improve trajectory forecasting compared to modular pipelines — a finding that motivates our DAG-based integration of FastDriveCoT with MCTS.

VAD (Jiang et al., 2023) proposed a vectorized scene representation for end-to-end autonomous driving, enabling more structured reasoning over map elements and agent interactions. VAD's explicit encoding of road topology complements our approach by providing candidate action primitives that can serve as leaf nodes in our MCTS search tree.

Furthermore, there are many other articles that relate to optimizing test-time performance and efficiency, although not in the domain of autonomous VLA models specifically. Snell et al. aimed to optimize test-time compute primarily by adaptively allocating test-time compute given the current prompt. Test-time compute was found to change through two methods: searching through verifiers and modifying the model's proposal distribution. The verifiers were utilized to search for N different potential paths and choose the path that returned the highest reward; this method prioritized exploration of new paths that may yield better results. Modifying the model distribution involved the model revising its own response, allowing it to learn from past mistakes. By taking task difficulty into consideration, Snell et al. employed these two methods to optimize test-time compute.

Another popular method that aimed to improve efficiency and performance in complex tasks utilized Monte Carlo Tree Search (MCTS) in order to improve decision-making in LLMs. This method involved constructing a tree of decisions to make, traversing through specific paths, and returning the optimal decision and resulting return. This method balances the exploitation-exploration trade-off by deciding to either continue traversing the known currently optimal path, or exploring a new path that can potentially lead to a new optimal solution.

To address inference bottlenecks inherent in complex attention masks, we can use Block-Sparse FlashAttention (BSFA). BSFA optimizes attention mechanisms for sparse patterns by pruning value block memory transfers, presenting a pathway for accelerating dependency-graph attention masks.

---

## Approach

Our project's key contribution is a novel application and integration of existing search and adaptive-compute methods into the specific domain of autonomous VLA inference. We aim to first replicate the FastDriveCoT as a baseline method. We hope to reproduce similar results that were attained by FastDriveCoT. Specifically, we aim to achieve similar CoT generation time and meta action accuracy using Average Displacement Error (ADE) between predicted and actual trajectories.

Upon successfully replicating FastDriveCoT, we aim to incorporate MCTS into VLA models to improve its decision-making capabilities. We hope to improve performance of VLA models by utilizing MCTS to determine the optimal decision amongst many possible choices. In this task, CoT generation time and meta action accuracy will remain the evaluation metrics to directly compare the performance of MCTS to the original FastDriveCoT reasoning technique.

### MCTS Reward Model Formulation

Each MCTS node represents a partial trajectory prefix τ = (a₁, a₂, …, aₜ) of meta-actions generated by the VLA. The tree is expanded by sampling the next action aₜ₊₁ ~ π_θ(· | s, τ) from the Gemma 4 policy, where s is the current visual context. The value of a node is estimated by the UCB1 score used for tree traversal:

```
UCT(v) = Q(v) / N(v)  +  c · sqrt( ln N(parent(v)) / N(v) )
```

where Q(v) is the cumulative reward collected across all rollouts through node v, N(v) is the visit count, and c is an exploration constant tuned on the validation set.

The reward signal R(τ) is a weighted combination of three terms:

```
R(τ) = −λ₁ · ADE(τ, τ*)  +  λ₂ · IOU(a, a*)  −  λ₃ · Latency(τ)
```

- **ADE(τ, τ\*)** — Average Displacement Error between the predicted trajectory τ and the ground-truth τ\* over the 6.4-second horizon (lower is better, hence the negative sign).
- **IOU(a, a\*)** — Meta-action Intersection over Union between the predicted high-level action a and the ground-truth a\* (higher is better).
- **Latency(τ)** — Wall-clock CoT generation time, included to penalise trajectories that exceed the real-time latency budget.

The weights λ₁, λ₂, λ₃ will be tuned via grid search on the DriveLM validation split. During initial experiments we use a proxy reward of the model's own log-probability log π_θ(τ | s) in place of the ADE term, allowing offline search without ground-truth labels.

Finally, we optimize test-time computation with difficulty-adaptive computation allocation to manage the latency budget. Scenario difficulty will be estimated based on the number of critical objects (vehicles, pedestrians, or any other obstacles) that are in view at any given moment in time, derived directly from FastDriveCoT's enumeration stage. Routing simple tasks to the single-pass parallel decoding baseline, and complex scenarios triggering MCTS search. As a stretch goal, we will adapt BSFA to the DAG mask to further reduce latency.

---

## Expected Outcomes

We expect to be able to define a model framework to be able to trade off reasoning depth and real time latency requirements. We plan to successfully replicate FastDriveCoT's reported 3.1 to 4.1x CoT generation speedup on DriveLM as our baseline. We aim to keep the Average Displacement Error (ADE) between the predicted and ground-truth trajectories low and match reported meta-action IOU, while being able to reason for more complex actions like multi-actor traffic scenarios or potential driving routes. We hypothesize that incorporating deeper reasoning into VLA models via MCTS will improve decision making performance and trajectory ADE in complex scenarios. We hypothesize that we will be able to define a difficulty-adaptive heuristic that will efficiently make trade offs between latency (with fast CoT) and deeper trajectory exploration (with MCTS).

### Method Comparison

The table below summarises the planned experiments and the hypothesised effect on each metric relative to the sequential CoT baseline:

| Method | CoT Latency | Meta-Action IOU | Trajectory ADE | Notes |
|---|---|---|---|---|
| Sequential CoT (baseline) | 1× | — | — | Reference point |
| FastDriveCoT (parallel DAG) | ~0.25–0.32× | ≈ baseline | ≈ baseline | Target: 3.1–4.1× speedup |
| FastDriveCoT + MCTS | ~1–2× | ↑ (complex scenes) | ↓ (complex scenes) | Deeper search improves quality |
| + Difficulty-adaptive routing | ~0.3–0.5× | ↑ | ↓ | Fast on simple; MCTS on complex |
| + BSFA kernel (stretch) | < adaptive | ↑ | ↓ | Additional bandwidth savings |

### Ablation Studies

To isolate the contribution of each component we will run the following ablations:

| Ablation | Variable | Fixed |
|---|---|---|
| A1 — MCTS rollout depth | depth ∈ {1, 2, 4, 8} | λ₁=1, λ₂=0.5, λ₃=0.1, c=1.41 |
| A2 — Reward weights | (λ₁, λ₂, λ₃) grid search | depth=4, c=1.41 |
| A3 — UCT exploration constant | c ∈ {0.5, 1.0, 1.41, 2.0} | depth=4, λ from A2 |
| A4 — Difficulty threshold | simple_threshold ∈ {2, 3, 4, 5} | complex_threshold=7 |
| A5 — Reward proxy | log π_θ vs. validation ADE | depth=4, λ from A2 |

Each ablation is evaluated on the DriveLM validation split using all three primary metrics (latency, IOU, ADE). Statistical significance will be assessed with a paired t-test across scenes (p < 0.05).

---

## Plan

To ensure a balanced workload and avoid development bottlenecks, we have divided the project into two sequential phases. Replicating the FastDriveCoT baseline and then focusing on our test-time compute extensions. We intend to utilize pair programming for the critical integration phases.

**Daniel Rodriguez** will be responsible for engineering the custom causal attention masks and the parallel forward pass by the midway executive summary deadline. Then his responsibility shifts to system integration, batching optimizations, and, as a stretch goal, implementing the Block-Sparse FlashAttention (BSFA) kernel.

**Cynthia Han** will be responsible for architecting and implementing the DAG dependency graph scheduler by the midway executive summary deadline. She will then be responsible for the implementation of the Monte Carlo Tree Search (MCTS) algorithm and the associated reward model for trajectory evaluation.

**Lucy Chiu** will take responsibility in data processing the DriveLM dataset pipeline and authoring the evaluation scripts by the midway executive summary deadline. Then is responsible for implementing the difficulty-adaptive compute allocation strategy to route tasks based on scenario complexity.

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

- Tian et al. — DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models
- FastDriveCoT — Accelerating Chain-of-Thought Reasoning in Autonomous Driving VLAs
- Hu et al. (2023) — UniAD: Planning-oriented Autonomous Driving
- Jiang et al. (2023) — VAD: Vectorized Scene Representation for Efficient Autonomous Driving
- Snell et al. (2024) — Scaling LLM Test-Time Compute Optimally
- MCTS for LLMs — Monte Carlo Tree Search for Decision-Making in Language Models
- DriveLM — Driving with Graph Visual Question Answering
- BSFA — Block-Sparse FlashAttention for Efficient Sparse Attention
- Gemma 4 — Google DeepMind Open-Weights Multimodal Model
