import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from agent.callbacks import HumanFeedbackCallback, PlottingCallback
from stable_baselines3.common.env_util import make_vec_env

from env.arkanoid_env import ArkanoidEnv

def make_env():
    env = ArkanoidEnv(render_mode='rgb_array')
    env = Monitor(env)
    return env

def train(test_mode=False, pretrained_path=None):
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Number of parallel environments
    n_envs = 16 if not test_mode else 2
    
    # Create vectorized environment
    # SubprocVecEnv runs each env in a separate process
    env = make_vec_env(
        make_env, 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv if not test_mode else DummyVecEnv
    )
    
    # Stack frames to capture motion
    env = VecFrameStack(env, n_stack=4)
    
    # Ensure image is in channel-first format (C, H, W) for PyTorch
    env = VecTransposeImage(env)

    # Initialize agent
    if pretrained_path:
        print(f"Loading pretrained model from {pretrained_path}...")
        model = PPO.load(pretrained_path, env=env, device="cuda", verbose=1)
        # We need to reset learning rate and other params if we want to fine-tune properly
        # But PPO.load restores them. We might want to lower LR?
        model.learning_rate = 2.5e-4 # Ensure LR is set
    else:
        model = PPO(
            "CnnPolicy", 
            env, 
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=2.5e-4,          # Lowered learning rate for stability
            n_steps=1024 if not test_mode else 64,                # Significantly increased collection size
            batch_size=2048 if not test_mode else 64,             # Increased batch size for GPU efficiency (2048*16 = 32768 total steps)
            n_epochs=4,                  # Decreased epochs to reduce overfitting
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,              # Slightly increased clip range for faster learning
            ent_coef=0.02,    
            device="cuda",
            policy_kwargs=dict(net_arch=dict(pi=[1024, 1024], vf=[1024, 1024]))
        )
    
    # Train
    total_timesteps = 5000000000 if not test_mode else 1000 # Increase total steps as we run faster
    checkpoint_callback = CheckpointCallback(save_freq=100000 // n_envs, save_path='./models/', name_prefix='arkanoid_ppo')
    human_feedback_callback = HumanFeedbackCallback(data_dir="data", check_freq=1000, batch_size=64, epochs=5)
    plotting_callback = PlottingCallback(plot_freq=10000, save_path="./logs/training_plot.png")
    
    callbacks = CallbackList([checkpoint_callback, human_feedback_callback, plotting_callback])
    
    print(f"Starting training (test_mode={test_mode}, n_envs={n_envs})...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks if not test_mode else None)
    
    # Save final model
    model.save("arkanoid_ppo_final")
    print("Training finished and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode with fewer steps")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained model (e.g., models/arkanoid_bc_pretrained)")
    args = parser.parse_args()
    train(test_mode=args.test, pretrained_path=args.pretrained)
