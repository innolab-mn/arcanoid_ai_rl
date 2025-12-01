import numpy as np
import pygame
import pickle
import os
from env.arkanoid_env import ArkanoidEnv

def record_demo():
    env = ArkanoidEnv(render_mode='human')
    obs, _ = env.reset()
    
    observations = []
    actions = []
    total_reward = 0
    
    print("Start playing to record demonstrations!")
    print("Press ESC to stop recording and save.")
    
    running = True
    while running:
        # Handle events
        action = 0 # Stay
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_RIGHT]:
            action = 2
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Store observation and action
        # We need to store the observation BEFORE the step, and the action taken
        # obs is (84, 84, 1), we might want to save it as is
        observations.append(obs)
        actions.append(action)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward}")
            obs, _ = env.reset()
            total_reward = 0
            
    env.close()
    
    # Save data
    os.makedirs("data", exist_ok=True)
    import time
    timestamp = int(time.time())
    filename = f"data/demo_{timestamp}.pkl"
    print(f"Saving {len(observations)} samples to {filename}...")
    with open(filename, "wb") as f:
        pickle.dump({"observations": np.array(observations), "actions": np.array(actions)}, f)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    record_demo()
