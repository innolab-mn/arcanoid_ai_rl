# Implementation Plan - Arkanoid RL Optimization

## Goal
Optimize the Arkanoid RL agent by fixing flawed reward logic and tuning PPO hyperparameters for better stability and faster convergence.

## User Review Required
> [!IMPORTANT]
> **Reward Logic Change**: The previous logic gave a *positive* reward (up to +5) when the ball was lost if the paddle was close. This could incentivize losing the ball. I am changing this to a **penalty** (-10) with a small "consolation" bonus (max +2) for being close, ensuring the net result is always negative (-8 to -10).

## Proposed Changes

### Game Logic (`game/arkanoid.py`)
#### [MODIFY] [arkanoid.py](file:///home/tsogoo/work/game_rl/arkanoid/game/arkanoid.py)
- **Paddle Hit Reward**: Increase from +0.2 to **+1.0** (Stronger incentive to keep ball alive).
- **Brick Hit Reward**: Increase from +0.5 to **+5.0** (Stronger incentive to win).
- **Win Reward**: Set to fixed **+1000.0**.
- **Lose Life Penalty**: Set to **-10.0**.
- **Distance Bonus**: Keep logic but scale it so it reduces the penalty rather than overcoming it. Max bonus +2.0.
    - Net reward on death: -10 (base) + [0 to 2] (dist) = **-10 to -8**.

### Training Script (`agent/train.py`)
#### [MODIFY] [train.py](file:///home/tsogoo/work/game_rl/arkanoid/agent/train.py)
- **Learning Rate**: Increase to **2.5e-4** (Standard for Atari-style CNNs).
- **Entropy Coefficient**: Increase to **0.01** (More exploration).
- **n_steps**: Reduce to **1024** (Faster updates).
- **batch_size**: Set to **2048**.

## Verification Plan

### Automated Tests
- Run `python3 agent/train.py --test` to verify the training loop starts without errors and rewards are logged correctly.
- Run `python3 game/arkanoid.py` manually to check if the game feels right (though rewards are internal).

### Manual Verification
- Inspect the training logs (printed to stdout) to see if `ep_rew_mean` increases faster than before.
