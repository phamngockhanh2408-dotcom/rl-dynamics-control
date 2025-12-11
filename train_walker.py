import gymnasium as gym
from stable_baselines3 import PPO
import os
import time

MODEL_NAME = "ppo_walker2d_v5_stable"
TIMESTEPS = 500000

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_NAME)
LOG_DIR = os.path.join(CURRENT_DIR, "training_logs")

def train():
    """
    Training Phase: Train the agent using PPO algorithm.
    """
    print(f"[INFO] Starting training for {TIMESTEPS} timesteps...")
    print(f"[INFO] Logs will be saved to: {LOG_DIR}")
    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(MODEL_PATH)
    print(f"[SUCCESS] Model saved to: {MODEL_PATH}.zip")
    env.close()

def test():
    """
    Validation Phase: Load the trained model and render visualization.
    """
    print("[INFO] Loading model for testing/visualization...")
    env = gym.make("Walker2d-v5", render_mode="human")
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}.zip")
        print("Please run the training phase first.")
        return
    model = PPO.load(MODEL_PATH, env=env)
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.2)  
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    train()
    test()
