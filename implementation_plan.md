# Arkanoid RL Agent Implementation Plan

## Goal Description
Create a fully functional Arkanoid (Breakout) game using Pygame and train a Reinforcement Learning (RL) agent to play it using pixel-based observations.

## User Review Required
> [!NOTE]
> We will use `stable-baselines3` for the RL agent as it provides robust implementations of PPO and DQN.
> The game will be built with `pygame`.
> The environment will be wrapped using `gymnasium`.

## Proposed Changes

### Project Structure
```
arkanoid_rl/
├── game/
│   ├── arkanoid.py      # Core game logic
│   └── settings.py      # Game constants (screen size, colors, etc.)
├── env/
│   └── arkanoid_env.py  # Gymnasium wrapper
├── agent/
│   └── train.py         # Training script
├── requirements.txt
└── main.py              # Entry point for manual play or training
```

### Game Implementation (`game/`)
- **`settings.py`**: Define screen dimensions (e.g., 600x800), colors, brick layout.
- **`arkanoid.py`**:
    - `Game` class: Manages game loop, state, and rendering.
    - `Paddle`, `Ball`, `Brick` classes.
    - `step(action)` method: Updates game state based on action (Left, Right, None).
    - `get_state()` method: Returns the current screen pixel array.
    - `reset()` method: Resets the game.

### RL Environment (`env/`)
- **`arkanoid_env.py`**:
    - Inherits from `gymnasium.Env`.
    - `observation_space`: `Box(low=0, high=255, shape=(H, W, 3), dtype=uint8)`.
    - `action_space`: `Discrete(3)` (Left, Right, Stay).
    - `reset()`: Calls game reset.
    - `step()`: Calls game step, calculates reward (score increase), checks done condition.
    - `render()`: Renders the pygame window.

### RL Agent (`agent/`)
- **`train.py`**:
    - Sets up the environment.
    - Wraps env in `VecFrameStack` or similar if needed (though SB3 handles some of this, explicit stacking is good for motion).
    - Initializes PPO or DQN model.
    - Runs training loop.
    - Saves the model.

### Optimization
- **Observation Preprocessing**: Resize to 84x84 and grayscale in `arkanoid_env.py`.
- **Hyperparameter Tuning**: Increase `n_steps` to 2048, `batch_size` to 64.
- **Reward Shaping**: Increase rewards for hitting bricks and paddle.
- **CUDA Support**: Enable GPU acceleration.
- **Resource Optimization**: Use `SubprocVecEnv` for parallel environments.
- **Imitation Learning**: Implement `record_demo.py` and `pretrain.py` for Behavioral Cloning.
- **Interactive Training**: Implement `HumanFeedbackCallback` to enable online learning from human demonstrations.
- **Real-time Charts**: Implement `PlottingCallback` to visualize training metrics using Matplotlib.

## Verification Plan

### Automated Tests
- Run `main.py --mode manual` to verify game physics and playability.
- Run `check_env` from `stable_baselines3.common.env_checker` to validate the Gym environment.
- Run `train.py` for a short number of timesteps to ensure the training loop runs without errors.

### Manual Verification
- Watch the agent play after a few training iterations (or during training via render).
