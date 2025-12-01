from env.arkanoid_env import ArkanoidEnv
import numpy as np

def verify():
    print("Initializing environment...")
    env = ArkanoidEnv(render_mode='rgb_array')
    
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    assert obs.shape == (84, 84, 1)
    
    print("Stepping environment...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step result: reward={reward}, terminated={terminated}, info={info}")
    
    print("Environment verification successful!")
    env.close()

if __name__ == "__main__":
    verify()
