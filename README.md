# Arkanoid RL Agent

A reinforcement learning agent trained to play Arkanoid using PPO (Proximal Policy Optimization) with advanced features including imitation learning, interactive training, and real-time visualization.

## Features

- **PPO-based RL Agent**: Trained using Stable-Baselines3
- **Gymnasium Environment**: Custom environment wrapper for the Arkanoid game
- **CUDA Acceleration**: GPU-accelerated training and evaluation
- **Parallel Training**: 16 parallel environments for efficient data collection
- **Imitation Learning**: Pre-train the agent using human demonstrations (Behavioral Cloning)
- **Interactive Training**: Teach the agent in real-time while it's learning
- **Real-time Visualization**: Live charts showing training progress
- **Advanced Physics**: Paddle velocity affects ball trajectory for skilled play
- **Optimized Rewards**: Distance-based rewards and shaped incentives

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- pygame, gymnasium, stable-baselines3

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Play Manually
```bash
python3 main.py
```
Controls: Arrow keys (Left/Right) or mouse

### 2. Train the Agent
```bash
# Standard training (5M timesteps)
python3 agent/train.py

# Quick test run
python3 agent/train.py --test
```

### 3. Evaluate the Agent
```bash
python3 agent/evaluate.py
```

## Advanced Usage

### Imitation Learning Workflow

#### Step 1: Record Demonstrations
Play the game yourself to generate training data:
```bash
python3 record_demo.py
```
Play for a few minutes. Press **ESC** to save and exit.

#### Step 2: Pre-train the Agent
Train the agent to mimic your gameplay:
```bash
python3 pretrain.py
```
This saves a model to `models/arkanoid_bc_pretrained.zip`.

#### Step 3: Fine-tune with RL
Load the pre-trained model and continue training with PPO:
```bash
python3 agent/train.py --pretrained models/arkanoid_bc_pretrained.zip
```

### Interactive Training (Human-in-the-Loop)

You can teach the agent **while it is training**:

1. Start training: `python3 agent/train.py`
2. Open a new terminal
3. Run `python3 record_demo.py` and play the game
4. When you finish (press ESC), the training script will automatically detect your new data and learn from it immediately

### Real-time Charts

A window will appear during training showing the **Mean Reward** over time, allowing you to monitor progress visually.

## Project Structure

```
arkanoid/
├── game/
│   ├── arkanoid.py       # Core game logic
│   └── settings.py       # Game constants
├── env/
│   └── arkanoid_env.py   # Gymnasium environment wrapper
├── agent/
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── callbacks.py      # Custom callbacks (HumanFeedback, Plotting)
├── main.py               # Manual play script
├── record_demo.py        # Record human demonstrations
├── pretrain.py           # Behavioral cloning pre-training
├── requirements.txt      # Python dependencies
├── walkthrough.md        # Detailed walkthrough
├── implementation_plan.md # Technical implementation details
└── task.md               # Task checklist
```

## Optimization Details

- **Observation Space**: Resized to 84x84 grayscale, stacked 4 frames for motion detection
- **Hyperparameters**: Tuned PPO parameters (n_steps=256, batch_size=256, n_epochs=10)
- **Rewards**: 
  - +2 for hitting the paddle
  - +0.5 per brick destroyed
  - Distance-based reward when ball is lost (encourages tracking)
  - +100 for winning
- **Physics**: Paddle velocity affects ball direction for skilled control
- **Parallel Training**: 16 environments running simultaneously via `SubprocVecEnv`

## Tips for Best Results

1. **Pre-training helps**: Record 5-10 minutes of gameplay for better initial performance
2. **Monitor charts**: Watch the real-time reward plot to see if training is progressing
3. **Interactive corrections**: If the agent develops bad habits, record demonstrations showing the correct behavior
4. **GPU recommended**: Training is significantly faster with CUDA

## Technical Details

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: CNN-based (processes pixel observations)
- **Framework**: Stable-Baselines3
- **Observation**: 84x84 grayscale, 4-frame stack
- **Action Space**: Discrete(3) - Stay, Left, Right
- **Training Steps**: 5M (customizable)

## License

This project is for educational purposes.

## Acknowledgments

Built with:
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://gymnasium.farama.org/)
- [Pygame](https://www.pygame.org/)
