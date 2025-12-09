import gymnasium as gym
from stable_baselines3 import PPO
import os
import time

# --- CONFIGURATION ---
MODEL_NAME = "ppo_walker2d_v5_stable"
TIMESTEPS = 500000

# Get current directory to save model/logs in the same folder as this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_NAME)
LOG_DIR = os.path.join(CURRENT_DIR, "training_logs")

def train():
    """
    Training Phase: Train the agent using PPO algorithm.
    """
    print(f"[INFO] Starting training for {TIMESTEPS} timesteps...")
    print(f"[INFO] Logs will be saved to: {LOG_DIR}")

    # Initialize environment (Headless mode for speed)
    env = gym.make("Walker2d-v5", render_mode="rgb_array")

    # Initialize the Agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

    # Start learning
    model.learn(total_timesteps=TIMESTEPS)

    # Save the model
    model.save(MODEL_PATH)
    print(f"[SUCCESS] Model saved to: {MODEL_PATH}.zip")
    
    env.close()

def test():
    """
    Validation Phase: Load the trained model and render visualization.
    """
    print("[INFO] Loading model for testing/visualization...")

    # Initialize environment with human rendering
    env = gym.make("Walker2d-v5", render_mode="human")

    # Check if model exists
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}.zip")
        print("Please run the training phase first.")
        return

    # Load the trained agent
    model = PPO.load(MODEL_PATH, env=env)

    # Main control loop
    obs, _ = env.reset()
    while True:
        # Predict action based on observation (Deterministic)
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        time.sleep(0.2)  # Control speed of rendering
        
        # Reset if episode ends
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    # Uncomment 'train()' to retrain the model. 
    # Comment it out if you only want to test an existing model.
    train()
    
    # Visualization
    test()