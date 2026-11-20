# Critical node identification in simulated aerial swarms

AE5510 course project (Problem 2), K. Sai Siddhartha Bajrang (AE22B002).

A graph neural network (edge-aware GIN layers, global feature attention, per-timestep swarm readout) that
**ranks** the agents of a swarm by `criticality_score` and **classifies** `is_critical`, trained with a
differentiable Spearman loss plus a class-weighted cross-entropy and evaluated on swarm sizes, attrition
levels and obstacle densities held out of training.

## Abstract

In a UAV swarm flying through cluttered airspace, losing some agents splits the communication network much more than losing others. If the swarm can find these critical agents in real time, it can protect them, reposition around them or plan for losing them. Exact critical-node detection is a combinatorial problem and has to be solved again every time the topology changes, which is too slow for a moving swarm. This repository has the code and report for an AE5510 course project at IIT Madras. The model learns to rank agents by criticality directly from the swarm's state at each timestep.

Each timestep of a simulated mission is a communication graph. Each node has 34 features covering the agent's motion, its local geometry and the state of the whole swarm. The model first applies learned feature attention. Three GINE layers (an edge-aware Graph Isomorphism Network) follow, with a mean-pooled swarm-context readout. A small MLP head then gives a criticality score for every agent at every timestep. The model has 17,174 parameters and a latent width of d = 12. The loss combines a differentiable soft-Spearman ranking term with class-weighted binary cross-entropy. This makes the scores useful both for ranking agents and for flagging critical ones with a threshold.

Evaluation uses a hard out-of-distribution split. Test swarms differ from the training swarms in size (52, 68 or 75 agents), attrition level or obstacle density. On this split the model reaches a per-timestep Spearman correlation of 0.816, NDCG@10 of 0.809, AUPRC of 0.873 and ROC-AUC of 0.900. Averaging each agent's scores over the mission raises Spearman to 0.863 and AUPRC to 0.933. Ranking improves as the mission goes on, with Spearman rising from 0.59 to 0.92. It is also stable from one step to the next (consistency 0.993, against 0.996 for the ground truth). It holds up on swarm sizes not seen in training, and it is unaffected by noise in the dynamics, dropping only under large position noise. Ablations over latent width, feature set and loss show that the BCE term mainly helps AUPRC (0.801 → 0.877). Logistic-regression and gradient-boosting baselines are included for comparison.

The full report is in [`report/report.pdf`](report/report.pdf).

## Contents

| Folder | What it is |
|---|---|
| `per_timestep/` | Final model (report, main results): one score per agent **per timestep**. Five experiments as `.py` scripts and single-cell notebooks. |
| `agent_level/` | Model presented at the final review: per-step latent states pooled over time by temporal attention, one score per agent (report, Appendix A). |
| `report/` | Final report: PDF, LaTeX source and figures. |

Each experiment is self-contained (one file, one cell) and writes all tables, figures and the model checkpoint
to its own folder (e.g. `exp4_in_depth_analysis_per_timestep/`).

| Experiment | Purpose |
|---|---|
| `exp1_dimension_analysis` | latent size d = 2, 4, 8, 12, 16, 20 |
| `exp2_relative_vs_absolute` | 34 inputs (with within-swarm z-scores) vs 23 absolute inputs |
| `exp3_bce_ablation` | loss with and without the classification term (lambda = 0.5 vs 0) |
| `exp4_in_depth_analysis` | final model: metrics, confusion matrix, feature attention, performance by swarm size / attrition / obstacle density / mission time |
| `exp5_robustness` | Gaussian noise on test positions, velocities or both |

## Setup

```bash
bash setup_env.sh            # creates .venv_critical with CPU-only PyTorch and a Jupyter kernel
# or: pip install -r requirements.txt  (install torch from https://download.pytorch.org/whl/cpu)
```

## Running

The dataset (`swarm_criticality_dataset_100_scenarios.csv`, one row per agent per timestep) is not included.
Point the scripts to it with an environment variable:

```bash
export SWARM_CSV=/path/to/swarm_criticality_dataset_100_scenarios.csv
cd per_timestep
python exp4_in_depth_analysis_per_timestep.py
```

or open the matching notebook and run its single cell. Settings (latent size `D`, `BCE_WEIGHT`, `PATIENCE`,
`TIME_STRIDE`, ...) are at the top of each file. Training runs on CPU; the main settings used in the report are
d = 12, lambda = 0.5, patience 500 epochs, every 3rd timestep.
