import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from auv_env import AUVEnvironment  # Make sure auv_env.py is in the same folder or installed as a module

def train(num_obstacles=3):
    # Create the environment with the specified number of obstacles
    env = AUVEnvironment(num_obstacles=num_obstacles)

    # Define the model (SAC)
    model = SAC(
        policy="MlpPolicy",  # MLP Policy for SAC
        env=env,
        verbose=1,
        tensorboard_log="./sac_auv_tensorboard/",  # Path to save tensorboard logs
    )

    # Optional checkpoint callback: saves every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save model every 10000 timesteps
        save_path="./checkpoints/",  # Path to save the model checkpoints
        name_prefix="auv_sac"  # Prefix for the checkpoint files
    )

    # Start training the model
    model.learn(
        total_timesteps=10000,  # Total number of training timesteps
        callback=checkpoint_callback  # Add checkpoint callback to save progress
    )

    # Save the final trained model
    model.save("auv_sac_model")

    print("✅ Training complete. Model saved as 'auv_sac_model'.")

if __name__ == "__main__":
    # Example usage: train the model with 5 obstacles
    train(num_obstacles=5)  # Specify the number of obstacles here (e.g., 5)
