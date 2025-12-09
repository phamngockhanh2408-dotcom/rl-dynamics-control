import gymnasium as gym
from stable_baselines3 import PPO
import os

# --- CONFIGURATION ---
ENV_ID = "InvertedDoublePendulum-v4"
MODEL_NAME = "ppo_double_pendulum"
TIMESTEPS = 100000 

# Path handling: Save model/logs in the same directory as this script
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

    # Initialize environment (Headless mode)
    env = gym.make(ENV_ID, render_mode="rgb_array")

    # Initialize PPO Agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

    # Execute training
    model.learn(total_timesteps=TIMESTEPS)

    # Save model
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

    # Initialize environment with Human Rendering
    env = gym.make(ENV_ID, render_mode="human")

    # Verify model existence
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}.zip")
        print("Please run the training phase first.")
        return

    # Load trained model
    model = PPO.load(MODEL_PATH, env=env)

    # Control loop
    obs, _ = env.reset()
    while True:
        # Predict action (Deterministic for evaluation)
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render frame
        env.render()

        # Reset if pendulum falls or episode ends
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    # Phase 1: Train
    train()
    
    # Phase 2: Test / Visualize
    test()