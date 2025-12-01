import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3.common.callbacks import BaseCallback

class HumanFeedbackCallback(BaseCallback):
    """
    Callback for Interactive Imitation Learning.
    Checks for new demonstration files and trains the agent on them.
    """
    def __init__(self, data_dir="data", check_freq=1000, batch_size=64, epochs=5, verbose=1):
        super().__init__(verbose)
        self.data_dir = data_dir
        self.check_freq = check_freq
        self.batch_size = batch_size
        self.epochs = epochs
        self.processed_files = set()
        self.criterion = nn.CrossEntropyLoss()
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self._check_and_train()
        return True

    def _check_and_train(self):
        # Find new files
        all_files = glob.glob(os.path.join(self.data_dir, "demo_*.pkl"))
        new_files = [f for f in all_files if f not in self.processed_files]
        
        if not new_files:
            return

        if self.verbose > 0:
            print(f"Found {len(new_files)} new demonstration files: {new_files}")

        # Load data
        observations = []
        actions = []
        
        for f_path in new_files:
            try:
                with open(f_path, "rb") as f:
                    data = pickle.load(f)
                    observations.append(data["observations"])
                    actions.append(data["actions"])
                self.processed_files.add(f_path)
            except Exception as e:
                print(f"Error loading {f_path}: {e}")
                
        if not observations:
            return

        # Concatenate data
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        
        # Preprocess
        # Observations are (N, 84, 84, 1). SB3 CnnPolicy expects (N, C, H, W)
        observations = np.transpose(observations, (0, 3, 1, 2))
        
        obs_tensor = torch.tensor(observations, dtype=torch.float32) / 255.0
        action_tensor = torch.tensor(actions, dtype=torch.long)
        
        dataset = TensorDataset(obs_tensor, action_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train
        policy = self.model.policy
        optimizer = optim.Adam(policy.parameters(), lr=3e-4) # Use a separate optimizer or the model's?
        # Using a separate optimizer is safer to avoid messing up PPO's internal state (e.g. Adam statistics)
        # However, we are modifying the weights, so PPO's optimizer might get confused.
        # But for simple BC updates, it's usually fine.
        
        policy.train()
        device = self.model.device
        
        if self.verbose > 0:
            print(f"Training on {len(observations)} human samples for {self.epochs} epochs...")
            
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_obs, batch_actions in dataloader:
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)
                
                optimizer.zero_grad()
                
                features = policy.extract_features(batch_obs)
                latent_pi = policy.mlp_extractor.forward_actor(features)
                logits = policy.action_net(latent_pi)
                
                loss = self.criterion(logits, batch_actions)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if self.verbose > 0:
                print(f"BC Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(dataloader):.4f}")
                
        if self.verbose > 0:
            print("Human feedback training complete.")

import matplotlib.pyplot as plt

class PlottingCallback(BaseCallback):
    """
    Callback for plotting training metrics in real-time.
    """
    def __init__(self, plot_freq=1000, verbose=1):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.rewards = []
        self.timesteps = []
        self.fig, self.ax = plt.subplots()
        plt.ion() # Interactive mode
        self.ax.set_title("Training Mean Reward")
        self.ax.set_xlabel("Timesteps")
        self.ax.set_ylabel("Mean Reward")

    def _on_step(self) -> bool:
        # Check for episode end
        for info in self.locals['infos']:
            if 'episode' in info:
                self.rewards.append(info['episode']['r'])
                self.timesteps.append(self.num_timesteps)
        
        if self.n_calls % self.plot_freq == 0 and len(self.rewards) > 0:
            self._update_plot()
            
        return True

    def _update_plot(self):
        self.ax.clear()
        self.ax.set_title("Training Mean Reward")
        self.ax.set_xlabel("Timesteps")
        self.ax.set_ylabel("Reward")
        
        # Plot raw rewards
        self.ax.plot(self.timesteps, self.rewards, label="Episode Reward", alpha=0.3)
        
        # Plot moving average
        if len(self.rewards) > 10:
            window = 10
            moving_avg = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
            # Adjust timesteps for valid convolution
            valid_timesteps = self.timesteps[window-1:]
            self.ax.plot(valid_timesteps, moving_avg, label="Moving Avg (10)", color='red')
            
        self.ax.legend()
        plt.draw()
        plt.pause(0.001)
