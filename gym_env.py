"""define a gymnasium environment for RL that has dim of data.ctrl as the action dim and the current position and velocity of each joint as obs (size model.nq + model.nv). This environment should introduce sensor noise + control delay.

Initialize the environment with a trajectory object. Override the env.get_reward function that computes the reward by using trajectory.cost() + the sum of magnitude squared of the velocity + the magnitude squared of the difference between the previous and current action to ensure smoothness

Allow for batched and non-batched input
"""
import gymnasium as gym
import numpy as np
import mujoco
from collections import deque
from trajectories import Trajectory

class FrankaEnv(gym.Env):
    def __init__(self, trajectory: Trajectory, xml_path="franka_emika_panda/panda_nohand.xml", delay_steps=2, noise_std=0.01):
        super().__init__()
        
        self.trajectory = trajectory

        # Loss constants
        self.cv = 0
        self.ca = 0
        
        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Setup spaces
        # Action space: dim of data.ctrl
        self.action_space = gym.spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0], 
            high=self.model.actuator_ctrlrange[:, 1], 
            dtype=np.float32
        )
        
        # Observation space: qpos (nq) + qvel (nv)
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Delay and noise parameters
        self.delay_steps = delay_steps
        self.noise_std = noise_std
        
        # Buffer for control delay
        self.action_buffer = deque(maxlen=self.delay_steps + 1)
        self.prev_action = np.zeros(self.action_space.shape)
        
    def _get_obs(self):
        # Get ground truth state
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Introduce sensor noise
        if self.noise_std > 0:
            qpos += np.random.normal(0, self.noise_std, size=qpos.shape)
            qvel += np.random.normal(0, self.noise_std, size=qvel.shape)
            
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def step(self, action):
        # Add current action to buffer
        self.action_buffer.append(action)
        
        # Pop delayed action (or use current if buffer isn't full yet)
        if len(self.action_buffer) > self.delay_steps:
            delayed_action = self.action_buffer.popleft()
        else:
            delayed_action = self.action_buffer[0]
            
        # Apply action and step simulation
        self.data.ctrl[:] = delayed_action
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self.get_reward(action)
        self.prev_action = np.copy(action)
        terminated = False
        truncated = False
        info = {}
        
        return obs, reward, terminated, truncated, info

    def get_reward(self, action):
        # 1. Trajectory cost
        # Find the end effector position from the 'attachment_site' site
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        eff_pos = self.data.site_xpos[site_id]
        
        t0 = self.data.time
        traj_cost = self.trajectory.cost(eff_pos, t0)
        
        # 2. Velocity penalty
        vel_penalty = np.sum(np.square(self.data.qvel))
        
        # 3. Action smoothness penalty
        action_penalty = np.sum(np.square(action - self.prev_action))
        
        # Using negative cost as the reward directly.
        reward = -(traj_cost + self.cv * vel_penalty + self.ca * action_penalty)
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Clear action buffer and populate with zeros
        self.action_buffer.clear()
        for _ in range(self.delay_steps + 1):
            self.action_buffer.append(np.zeros_like(self.action_space.sample()))
        
        self.prev_action = np.zeros(self.action_space.shape)
            
        obs = self._get_obs()
        info = {}
        
        return obs, info


