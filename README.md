# rl-dynamics-control
Deep Reinforcement Learning for Underactuated Robotics Systems (Walker2d &amp; Double Pendulum).
# Reinforcement Learning for Physical Control Systems

## 1. Project Overview
This repository demonstrates the application of **Deep Reinforcement Learning (PPO)** to solve complex control problems in continuous action spaces. 
The project focuses on two classic non-linear dynamics problems:
1.  **Bipedal Locomotion (Walker2d-v5):** Learning stable gait generation.
2.  **Inverted Double Pendulum:** Stabilization of a chaotic underactuated system.

**Tech Stack:** Python, Gymnasium (MuJoCo Physics Engine), Stable Baselines3, PyTorch.

---

## 2. Project A: Bipedal Locomotion (Walker2d)

### Objective
To train a 2D bipedal robot to walk forward while maintaining torso stability and energy efficiency.

### Engineering Approach
* **Physics Model:** Based on **MuJoCo** engine. The robot consists of 7 links and 6 actuated joints (hips, knees, ankles).
* **Algorithm:** Proximal Policy Optimization (PPO).
* **Reward Shaping (Key Contribution):** Instead of using the default sparse reward, I optimized the reward function to prioritize stability:
    * `forward_reward_weight`: Tuned to balance speed vs. stability.
    * `healthy_reward`: Increased to encourage the agent to maintain an upright center of mass (CoM).
    * `ctrl_cost`: Penalized high-torque actions to simulate energy-efficient actuators.

### Result (Demo)
*The agent achieved a stable walking gait after 500,000 timesteps.*

https://github.com/phamngockhanh2408-dotcom/rl-dynamics-control/blob/main/Walker%202D.mp4

---

## 3. Project B: Double Inverted Pendulum Stabilization

### Objective
To balance a double inverted pendulum on a cart. This is a classic **underactuated system** problem (fewer actuators than degrees of freedom).

### Challenges
* **Non-linearity:** The system is highly sensitive to initial conditions (Chaos theory).
* **Precision:** Requires high-frequency control updates to maintain the upright position.

### Result (Demo)
*The agent successfully balances the pendulum indefinitely.*

https://github.com/phamngockhanh2408-dotcom/rl-dynamics-control/blob/main/DoublePendulum.mp4

---

## 4. How to Run
1. Install dependencies:
   ```bash
   pip install gymnasium[mujoco] stable-baselines3 shimmy
