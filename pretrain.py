import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3 import PPO
from env.arkanoid_env import ArkanoidEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

def pretrain():
    # Load data
    try:
        with open("data/demonstrations.pkl", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("No demonstration data found. Run record_demo.py first.")
        return

    observations = data["observations"]
    actions = data["actions"]
    
    print(f"Loaded {len(observations)} samples.")
    
    # Preprocess data
    # Observations are (N, 84, 84, 1).
    # We need to stack 4 frames to match VecFrameStack
    # We will slide a window of 4 frames
    
    N, H, W, C = observations.shape
    stacked_obs = []
    stacked_actions = []
    
    # We need at least 4 frames
    if N < 4:
        print("Not enough data to stack frames.")
        return

    for i in range(3, N):
        # Stack frames i-3, i-2, i-1, i
        # Each frame is (84, 84, 1)
        # We want (4, 84, 84)
        # But wait, SB3 VecFrameStack stacks on channel dimension?
        # Let's check. Gym wrappers usually stack on last dimension (H, W, C*4).
        # But we use VecTransposeImage in train.py which converts to (C*4, H, W).
        # So we should prepare (4, 84, 84).
        
        # Let's verify how VecFrameStack works.
        # It stacks on the last dimension. So (H, W, C) -> (H, W, C*4).
        # Then VecTransposeImage converts to (C*4, H, W).
        
        # So we should first stack on last dimension: (84, 84, 4)
        frames = observations[i-3:i+1] # Shape (4, 84, 84, 1)
        stack = np.concatenate(frames, axis=-1) # Shape (84, 84, 4)
        
        stacked_obs.append(stack)
        stacked_actions.append(actions[i])
        
    stacked_obs = np.array(stacked_obs)
    stacked_actions = np.array(stacked_actions)
    
    print(f"Stacked data shape: {stacked_obs.shape}")
    
    # Now transpose to (N, C, H, W) for PyTorch/SB3
    # stacked_obs is (N, 84, 84, 4) -> (N, 4, 84, 84)
    stacked_obs = np.transpose(stacked_obs, (0, 3, 1, 2))
    
    # Convert to tensors
    obs_tensor = torch.tensor(stacked_obs, dtype=torch.float32) / 255.0 # Normalize
    action_tensor = torch.tensor(stacked_actions, dtype=torch.long)
    
    dataset = TensorDataset(obs_tensor, action_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Create dummy env for model initialization
    # We need to match the structure in train.py: VecFrameStack(DummyVecEnv(ArkanoidEnv))
    # But we can just create a dummy env with the correct observation space
    from gymnasium import spaces
    
    # We need to trick SB3 into thinking it has the right env
    # We can use the same make_env and wrappers as train.py
    from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
    
    env = DummyVecEnv([lambda: ArkanoidEnv(render_mode='rgb_array')])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    # Initialize PPO model
    model = PPO("CnnPolicy", env, verbose=1, device="cuda")
    
    # We will train the policy network (actor)
    # Access the policy
    policy = model.policy.to("cuda")
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting pre-training...")
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        for batch_obs, batch_actions in dataloader:
            batch_obs = batch_obs.to("cuda")
            batch_actions = batch_actions.to("cuda")
            
            optimizer.zero_grad()
            
            # Forward pass
            # policy.get_distribution(obs) returns a distribution
            # We can get logits from the distribution or just use the features extractor + action net
            # SB3 policy forward returns actions, values, log_probs. We want logits/probs.
            # Let's use evaluate_actions? No, that gives log_prob.
            # Let's look at SB3 ActorCriticPolicy.
            # It has extract_features and action_net.
            
            features = policy.extract_features(batch_obs)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)
            
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
        
    # Save the pretrained model
    model.save("models/arkanoid_bc_pretrained")
    print("Pre-trained model saved to models/arkanoid_bc_pretrained.zip")

if __name__ == "__main__":
    pretrain()
