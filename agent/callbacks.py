import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class HumanFeedbackCallback(BaseCallback):
    """
    Callback for Interactive Imitation Learning.
    Checks for new demonstration files and trains the agent on them using Behavioral Cloning (BC).
    """
    def __init__(self, data_dir="data", check_freq=1000, batch_size=64, epochs=5, n_stack=4, verbose=1):
        super().__init__(verbose)
        self.data_dir = data_dir
        self.check_freq = check_freq
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_stack = n_stack
        self.processed_files = set()
        self.criterion = nn.CrossEntropyLoss()
        
    def _frame_stack_demonstrations(self, observations, actions):
        """
        Stacks sequential single-channel observations to match the VecFrameStack output.
        Input observations shape: (N, H, W, 1) where N is the total number of steps.
        Output observations shape: (N_stacked, H, W, n_stack)
        
        The first n_stack - 1 actions and observations are dropped because a full stack is not available.
        """
        N = observations.shape[0]
        if N < self.n_stack:
            if self.verbose > 0:
                 print(f"Warning: Only {N} frames available, need at least {self.n_stack} for stacking. Skipping batch.")
            return np.array([]), np.array([])
            
        stacked_obs = []
        
        for i in range(self.n_stack - 1, N):
            # Stack the n_stack frames from i - n_stack + 1 up to i
            # We index with [:, :, :, 0] to remove the single channel dimension (1)
            stack = observations[i - self.n_stack + 1 : i + 1, :, :, 0]
            # stack shape is now (n_stack, H, W). 
            # Transpose: (n_stack, H, W) -> (H, W, n_stack) to match Gym's convention
            stack = np.transpose(stack, (1, 2, 0)) 
            stacked_obs.append(stack)

        # The actions tensor corresponds to the actions taken after observing the stack ending at index i
        stacked_actions = actions[self.n_stack - 1 : N]

        return np.array(stacked_obs, dtype=observations.dtype), stacked_actions

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

        # Load raw data
        all_observations = []
        all_actions = []
        
        for f_path in new_files:
            try:
                with open(f_path, "rb") as f:
                    data = pickle.load(f)
                    all_observations.append(data["observations"])
                    all_actions.append(data["actions"])
                self.processed_files.add(f_path)
            except Exception as e:
                print(f"Error loading {f_path}: {e}")
                
        if not all_observations:
            return

        # Concatenate raw data
        raw_observations = np.concatenate(all_observations, axis=0)
        raw_actions = np.concatenate(all_actions, axis=0)

        # Apply Frame Stacking (The essential fix)
        stacked_observations, actions = self._frame_stack_demonstrations(raw_observations, raw_actions)
        
        if len(stacked_observations) == 0:
             return

        # Preprocess for PyTorch
        # Stacked observations are (N_stacked, H, W, C=4).
        # Transpose to (N_stacked, C, H, W) for PyTorch CNN:
        observations = np.transpose(stacked_observations, (0, 3, 1, 2))
        
        # Convert to tensors and normalize
        obs_tensor = torch.tensor(observations, dtype=torch.float32) / 255.0
        action_tensor = torch.tensor(actions, dtype=torch.long)
        
        dataset = TensorDataset(obs_tensor, action_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train
        policy = self.model.policy
        # Use a new optimizer for BC updates
        optimizer = optim.Adam(policy.parameters(), lr=3e-4) 
        
        policy.train()
        device = self.model.device
        
        if self.verbose > 0:
            print(f"Training on {len(stacked_observations)} human samples for {self.epochs} epochs...")
            
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_obs, batch_actions in dataloader:
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass through the policy
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

# Assuming BaseCallback is imported from stable_baselines3.common.callbacks

class PlottingCallback(BaseCallback):
    """
    Callback for plotting training metrics in real-time and saving the plot image.
    """
    def __init__(self, plot_freq=1000, save_path="./plots/training_reward.png", verbose=1):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.save_path = save_path
        self.rewards = []
        self.timesteps = []
        
        # Ensure the save directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Setup Matplotlib in interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6)) # Added figure size for better saved image
        plt.ion() 
        self.ax.set_title("Training Mean Reward")
        self.ax.set_xlabel("Timesteps")
        self.ax.set_ylabel("Mean Reward")

    def _on_step(self) -> bool:
        # Check for episode end in the info dictionary
        if self.locals.get('infos') is not None:
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
        
        # 🎯 FIX: Save the plot image
        try:
            self.fig.savefig(self.save_path)
            if self.verbose > 0:
                print(f"Plot saved to {self.save_path}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Error saving plot: {e}")
                
        # Redraw the plot and process events
        # plt.draw()
        plt.pause(0.001)