import gymnasium as gym
from stable_baselines3 import PPO
import os

ENV_ID = "InvertedDoublePendulum-v4"
MODEL_NAME = "ppo_double_pendulum"
TIMESTEPS = 100000 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_NAME)
LOG_DIR = os.path.join(CURRENT_DIR, "pendulum_logs")

def train():
    """
    Training Phase: Train the agent to balance the double pendulum.
    """
    print(f"[INFO] Starting training for {ENV_ID}...")
    print(f"[INFO] Training duration: {TIMESTEPS} timesteps")
    print(f"[INFO] Tensorboard logs will be saved to: {LOG_DIR}")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(MODEL_PATH)
    print(f"[SUCCESS] Model saved to: {MODEL_PATH}.zip")
    env.close()

def test():
    """
    Testing Phase: Load model and visualize the control policy.
    """
    print("-------------------------------------------------")
    print(f"[INFO] Visualizing {ENV_ID} agent... (Press Ctrl+C to stop)")
    print("-------------------------------------------------")
    env = gym.make(ENV_ID, render_mode="human")
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
        if terminated or truncated:
            obs, _ = env.reset()
if __name__ == "__main__":
    train()
    test()

