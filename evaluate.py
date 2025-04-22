import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from auv_env import AUVEnvironment  # Make sure auv_env.py is in the same folder or installed as a module

def evaluate(model_path="auv_sac_model.zip", episodes=5, render=False, max_steps=200000, num_obstacles=5):
    # Create the environment with the specified number of obstacles
    env = AUVEnvironment(num_obstacles=num_obstacles)

    # Load the trained model
    model = SAC.load(model_path)

    all_trajectories = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0
        trajectory = [env.position.copy()]

        while not done and step < max_steps:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            step += 1
            total_reward += reward
            trajectory.append(env.position.copy())

            if render:
                print(f"[Ep {ep+1} | Step {step}] Pos: {env.position} | Reward: {reward:.2f}")
                env.render()

        all_trajectories.append(np.array(trajectory))
        print(f"✅ Episode {ep+1} finished | Steps: {step} | Total Reward: {total_reward:.2f}")

    # Plot everything
    plt.figure(figsize=(10, 10))

    for i, traj in enumerate(all_trajectories):
        plt.plot(traj[:, 0], traj[:, 1], marker='o', linewidth=1, markersize=2, label=f"Episode {i+1}")
        plt.scatter(traj[0, 0], traj[0, 1], color='green', marker='D', s=80)  # Start
        plt.scatter(traj[-1, 0], traj[-1, 1], color='blue', marker='x', s=80)  # End point (in case goal not reached)

    # Draw goal
    if hasattr(env, "target"):
        plt.scatter(*env.target, color='red', marker='*', s=200, label='Goal')

    # Draw obstacles
    for (ox, oy, r) in env.obstacles:
        circle = plt.Circle((ox, oy), r, color='gray', alpha=0.4)
        plt.gca().add_patch(circle)

    plt.title("AUV Trajectories in 2D Environment")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("auv_trajectory_plot.png")
    print("📊 Trajectory plot saved as 'auv_trajectory_plot.png'")
    plt.close()
    env.close()

if __name__ == "__main__":
    # Example usage: evaluate a model with 5 obstacles
    evaluate(model_path="auv_sac_model.zip", episodes=5, render=True, num_obstacles=5)
