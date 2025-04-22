import numpy as np
import gym
from gym import spaces

class AUVEnvironment(gym.Env):
    def __init__(self, num_obstacles=10):
        super(AUVEnvironment, self).__init__()
        
        self.map_size = 10.0  # 10x10 area
        self.dt = 0.1  # time step
        self.max_steps = 10000  # Prevent infinite episodes

        # Target is fixed and should NOT change during training
        self.target = np.array([9.0, 9.0])

        # Action space: [dx, dy] (velocity components)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: [x, y, vx, vy]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Number of obstacles and their placements
        self.num_obstacles = num_obstacles
        self.obstacles = self.generate_random_obstacles(self.num_obstacles)

        self.reset()

    def generate_random_obstacles(self, num_obstacles):
        """
        Generates random obstacles within the environment.
        Each obstacle has a random position and radius.
        """
        obstacles = []
        for _ in range(num_obstacles):
            ox = np.random.uniform(0.0, self.map_size)
            oy = np.random.uniform(0.0, self.map_size)
            r = np.random.uniform(0.3, 1.0)  # Random radius between 0.3 and 1.0
            obstacles.append((ox, oy, r))
        return obstacles

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.position = np.array([1.0, 1.0])  # Starting position of the AUV
        self.velocity = np.array([0.0, 0.0])  # Starting velocity of the AUV
        self.prev_distance = np.linalg.norm(self.position - self.target)  # Initial distance to target
        self.step_count = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        """
        Returns the current observation: [x, y, vx, vy]
        """
        return np.concatenate([self.position, self.velocity])

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {}

        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Normalize to ensure uniform thrust direction
        norm = np.linalg.norm(action)
        if norm > 1e-6:
            action = action / norm

        # === Obstacle avoidance logic ===
        repulsion_force = np.zeros(2)
        repulsion_threshold = 1.0  # Start repelling when closer than this
        repulsion_scale = 1.0      # Strength of repulsion

        for ox, oy, radius in self.obstacles:
            obstacle_pos = np.array([ox, oy])
            dist = np.linalg.norm(self.position - obstacle_pos)
            if dist < radius + repulsion_threshold:
                direction_away = self.position - obstacle_pos
                direction_away /= (np.linalg.norm(direction_away) + 1e-6)
                strength = repulsion_scale * (1.0 / (dist - radius + 1e-3))
                repulsion_force += strength * direction_away

        # Blend action and repulsion force
        action += repulsion_force
        if np.linalg.norm(action) > 1e-6:
            action = action / (np.linalg.norm(action) + 1e-6)

        self.velocity = action
        new_position = self.position + self.velocity * self.dt

        # Clip to environment bounds
        new_position = np.clip(new_position, 0, self.map_size)

        # Collision check
        collided = False
        for ox, oy, radius in self.obstacles:
            if np.linalg.norm(new_position - np.array([ox, oy])) < radius:
                collided = True
                break

        if collided:
            # Push back slightly from obstacle
            self.velocity = -self.velocity * 0.5
            self.position += self.velocity * self.dt
            reward = -5.0  # Smaller penalty to allow escaping
        else:
            self.position = new_position
            reward = 0.0

        # === Reward shaping ===
        dist_to_target = np.linalg.norm(self.position - self.target)
        progress = self.prev_distance - dist_to_target
        reward += 10.0 * progress

        to_target = self.target - self.position
        to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-8)
        velocity_norm = self.velocity / (np.linalg.norm(self.velocity) + 1e-8)
        reward += 2.0 * np.dot(to_target_norm, velocity_norm)

        reward -= 0.05 * np.linalg.norm(self.velocity)  # Speed penalty
        reward -= 0.1  # Time penalty

        # Check if goal reached
        if dist_to_target < 0.5:
            reward += 100.0
            self.done = True

        self.prev_distance = dist_to_target
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, {}


    def render(self, mode='human'):
        """
        Renders the environment visually (using matplotlib).
        """
        import matplotlib.pyplot as plt
        plt.clf()
        ax = plt.gca()
        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)

        # Draw AUV
        plt.plot(self.position[0], self.position[1], 'bo', label='AUV')

        # Draw Goal
        plt.plot(self.target[0], self.target[1], 'r*', markersize=15, label='Goal')

        # Draw Obstacles
        for ox, oy, r in self.obstacles:
            circle = plt.Circle((ox, oy), r, color='gray', alpha=0.4)
            ax.add_patch(circle)

        plt.pause(0.001)
        plt.draw()
