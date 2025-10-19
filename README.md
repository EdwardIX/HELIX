# Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving

This repository is the source code for the paper: "Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving".

## Getting Started

### Environment Setup

1. **Create Conda Environment**
   ```bash
   conda create -n verl python==3.10
   conda activate verl
   ```

2. **Install PyTorch**
   ```bash
   pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
   ```

3. **Install Core Dependencies**
   ```bash
   pip install sglang==0.4.9
   pip install sgl-kernel==0.2.4
   pip install vllm==0.9.0
   ```

4. **Install Flash Attention**
   ```bash
   wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   ```

5. **Install Additional Packages**
   ```bash
   pip install transformers==4.53.3
   pip install hydra-core pandas tensordict torchdata codetiming peft matplotlib datasets wandb accelerate dill liger-kernel latex2sympy2_extended
   ```

### Quick Start Training
   ```bash
   cd recipe/algo_evolve
   bash training_script.sh experiment_name
   ```

## Repository Structure

```
verl/
├── recipe/                          # Training configurations and scripts
│   ├── algo_evolve/               # Evolutionary algorithm training
│   │   ├── dataset/               # Dataset handling and evolve algorithms
│   │   └── xxx.sh                 # Traing scripts for different tasks
│   └── direct_prompt/             # Direct prompt training
├── src/                           
│   ├── rewards/                   # Reward computation modules
│   ├── simplify/                  # Code normalization utilities
│   └── utils/                     # Utility functions and model management
```

## Algorithm Core

The evolutionary algorithm implementation is located in `recipe/algo_evolve/dataset/`:

### Key Components

1. **Dynamic RL Dataset** (`dynamic_rl_dataset.py`)
   - Handles dynamic dataset updates during training
   - Manages sample strategies and prompt sampling
   - Supports filtering and deduplication

2. **Prompt Samplers** (`prompt_sampler.py`)
   - `ComsolProcessor`: For physics simulation tasks
   - `MathProgramProcessor`: For mathematical programming problems
   - `AIProgramProcessor`: For AI coding tasks
   - `MathSymbolicRegressionProcessor`: For symbolic regression

3. **Sample Strategies** (`sample_strategy.py`)
   - `RandomSampleStrategy`: Random sampling
   - `TopKSampleStrategy`: Top-K sampling based on rewards
   - `NSGASampleStrategy`: Non-dominated sorting genetic algorithm
   - `TopKDivSampleStrategy`: Top-K with diversity consideration

4. **Embedding Modules** (`embedding_modules.py`)
   - Supports various embedding models for semantic similarity
   - Handles batch processing of text embeddings

5. **Grouped Dataset** (`grouped_dynamic_rl_dataset.py`)
   - Manages multiple data sources simultaneously
   - Provides parallel processing for different task types

## Reward Computation

The reward system is implemented in `src/rewards/` and consists of:

### Core Reward Classes

1. **Math Rewards** (`math/`)
   - `math_rewards.py`: Mathematical problem evaluation
   - `symbolic_regression.py`: Symbolic regression rewards
   - `ai_coding_rewards.py`: AI coding task evaluation
   - `external_evaluator_rewards.py`: External code execution

2. **COMSOL Rewards** (`comsol/`)
   - `comsol_reward_design.py`: Design optimization rewards

3. **Base Rewards** (`utils/`)
   - `basic_rewards.py`: basic rewards
   - `math_rewards.py`: Base mathematical reward computation

### Reward Servers

The framework supports multiple execution backends for output evaluation:

- **Local Server** (`utils/serv_local.py`): Multi-threaded local execution
- **SLURM Server** (`utils/serv_sbatch.py`): HPC cluster execution
- **bsub Server** (`utils/serv_bsub.py`): LSF cluster execution

### Reward Pipeline

1. **Input Processing**: Parse generated code and apply preprocessing
2. **Execution**: Run code in isolated environments
3. **Evaluation**: Compute task-specific rewards
4. **Feedback**: Provide structured feedback for next iteration
