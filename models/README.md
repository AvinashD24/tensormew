# TensorMew Pre-Trained Models & Checkpoints

This directory contains public model weights and checkpoints for PokéMind / TensorMew.

## Checkpoints

### 1. `deep_policy.pt` (~50.5 MB, 13,220,678 parameters)
- **Model Type:** `SpatialVisionPolicy` (`src/deep_policy.py`)
- **Architecture:**
  - Visual encoder: 4-stage convolutional residual trunk processing $144 \times 160$ PyBoy screen frames.
  - RAM encoder: Multi-layer perceptron processing 17 normalized game state features (coordinates, badges, party status, collision/dialogue state).
  - Fusion & Memory: 512-dim recurrent GRU cell with temporal hidden state tracking.
  - Policy Heads: Actor (5 categorical actions: Up, Down, Left, Right, A) and Critic (scalar value function estimate).
- **Training:** PPO with curriculum resets on RTX 4090.
- **Usage:**
  ```bash
  # Run live visual player using deep policy
  ./run_live.sh
  # or
  python -m tools.play_live --policy models/deep_policy.pt
  ```

### 2. `ppo_policy.npz` (~14.5 KB)
- **Model Type:** Linear / MLP Actor-Critic (`src/ppo.py`)
- **Usage:** Fast CPU training and testing for overworld segment traversal.
- **Usage:**
  ```bash
  python -m tools.run_segment --policy models/ppo_policy.npz --state new_bark_outside --steps 80 --window
  python -m tools.eval_exits --policy models/ppo_policy.npz --state new_bark_outside --episodes 8
  ```
