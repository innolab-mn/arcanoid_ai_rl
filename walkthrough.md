# Arkanoid RL Agent Walkthrough

I have successfully implemented the Arkanoid game and a Reinforcement Learning agent to play it.

## Project Structure
- `game/`: Contains the Arkanoid game logic (`arkanoid.py`) and settings (`settings.py`).
- `env/`: Contains the Gymnasium environment wrapper (`arkanoid_env.py`).
- `agent/`: Contains the training (`train.py`) and evaluation (`evaluate.py`) scripts.
- `main.py`: Script for manual play.
- `verify_env.py`: Script to verify the environment.

## How to Run

### 1. Manual Play
To play the game manually using arrow keys:
```bash
python3 main.py
```

### 2. Train the Agent
To train the PPO agent:
```bash
python3 agent/train.py
```
This will save checkpoints to `./models/` and the final model to `arkanoid_ppo_final.zip`.
Logs are saved to `./logs/` and can be viewed with TensorBoard.
Training is set to 1,000,000 timesteps for better performance.

### 3. Evaluate the Agent
To watch the trained agent play:
```bash
python3 agent/evaluate.py
```

## Optimization Details
- **Observation Space**: Resized to 84x84 and converted to grayscale to speed up training.
- **Hyperparameters**: Tuned PPO parameters (n_steps=2048, batch_size=64) for better stability.
- **Rewards**: Added reward for hitting the paddle (+0.5) and increased brick reward (+5). Added distance-based reward when ball is lost (closer = higher reward).
- **CUDA Support**: Configured agent to use CUDA for faster training and evaluation.
- **Parallel Environments**: Implemented `SubprocVecEnv` to run 16 environments in parallel, maximizing CPU and GPU utilization.
- **Imitation Learning**: Added support for recording human demos and pre-training the agent (Behavioral Cloning) to speed up learning.

## Imitation Learning Workflow

### 1. Record Demonstrations
Play the game yourself to generate training data:
```bash
python3 record_demo.py
```
Play for a few minutes to collect good data. Press ESC to save and exit.

### 2. Pre-train the Agent
Train the agent to mimic your gameplay:
```bash
python3 pretrain.py
```
This will save a model to `models/arkanoid_bc_pretrained.zip`.

### 3. Fine-tune with RL
Load the pre-trained model and continue training with PPO:
```bash
python3 agent/train.py --pretrained models/arkanoid_bc_pretrained.zip
```

### 4. Interactive Training (Human-in-the-Loop)
You can teach the agent *while* it is training!
1.  Start training (as above).
2.  Open a new terminal.
3.  Run `python3 record_demo.py` and play the game.
4.  When you finish (press ESC), the training script will automatically detect the new data and train on it immediately.

### 5. Real-time Charts
A window will appear during training showing the **Mean Reward** over time. This allows you to monitor progress visually.

## Verification Results

### Environment Verification
Ran `verify_env.py` to check observation shapes and step function.
- Observation shape: `(800, 600, 3)` (Correct)
- Step function returns valid reward and done flags.

### Training Verification
Ran a short training session (`python3 agent/train.py --test`) to ensure the training loop works.
- Training completed successfully.
- Model saved.

## Next Steps
- Train the agent for longer (e.g., 1M+ timesteps) to achieve good performance.
- Tune hyperparameters in `agent/train.py`.
- Experiment with DQN or other algorithms.
