import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from env.arkanoid_env import ArkanoidEnv
import time

def make_env():
    env = ArkanoidEnv(render_mode='human')
    return env

def evaluate():
    # Create environment
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    # Load model
    try:
        model = PPO.load("models/arkanoid_ppo_5000000_steps", device="cuda")
    except FileNotFoundError:
        print("Model not found. Please train the agent first.")
        return

    obs = env.reset()
    done = False
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.02) # Control playback speed

if __name__ == "__main__":
    evaluate()
