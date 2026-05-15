"""define a gymnasium environment for RL that has dim of data.ctrl as the action dim and the current position and velocity of each joint as obs (size model.nq + model.nv). This environment should introduce sensor noise + control delay.

Initialize the environment with a trajectory object. Override the env.get_reward function that computes the reward by using trajectory.cost() + the sum of magnitude squared of the velocity + the magnitude squared of the difference between the previous and current action to ensure smoothness

Allow for batched and non-batched input
"""
import gymnasium as gym
import os
import numpy as np
import mujoco
from collections import deque
from trajectories import Trajectory
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle

class FrankaEnv(MujocoEnv, EzPickle):
    def __init__(self, trajectory: Trajectory, xml_path="franka_emika_panda/panda_nohand.xml", frame_skip=1, delay_steps=2, noise_std=0.01, **kwargs):
        EzPickle.__init__(self, trajectory, xml_path, frame_skip, delay_steps, noise_std, **kwargs)
        
        self.trajectory = trajectory
        self.frames = np.array([])
        # Loss constants
        self.cv = 0.2
        
        # Delay and noise parameters
        self.delay_steps = delay_steps
        self.noise_std = noise_std
        
        # Buffer for control delay
        self.action_buffer = deque(maxlen=self.delay_steps + 1)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_xml_path = os.path.join(current_dir, xml_path)
        

        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(20,), 
            dtype=np.float32
        )
        MujocoEnv.__init__(self, model_path=full_xml_path, frame_skip=frame_skip, observation_space=self.observation_space, **kwargs)

        self.action_dim = 7
        self.observation_dim = 20
        
        
    def _get_obs(self):
        # Get ground truth state
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Introduce sensor noise
        if self.noise_std > 0:
            qpos += np.random.normal(0, self.noise_std, size=qpos.shape)
            qvel += np.random.normal(0, self.noise_std, size=qvel.shape)
            
        target_pos = self.trajectory.evaluate(self.data.time)

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        if site_id == -1:
            raise ValueError("attachment_site not found")
        eff_pos = self.data.site_xpos[site_id]

        return np.concatenate([qpos, qvel, target_pos, eff_pos])

    def step(self, action):
        # Add current action to buffer
        self.action_buffer.append(action)
        
        # Pop delayed action (or use current if buffer isn't full yet)
        if len(self.action_buffer) > self.delay_steps:
            delayed_action = self.action_buffer.popleft()
        else:
            delayed_action = self.action_buffer[0]
            
        # Apply action and step simulation using MujocoEnv's helper
        self.do_simulation(delayed_action, self.frame_skip)
        
        next_obs = self._get_obs()
        reward = self.get_reward(action, next_obs)
        self.prev_action = np.copy(action)
        terminated = self.is_terminated(action, next_obs)
        truncated = False
        info = {}
        
        if self.render_mode == "human":
            self.render()

        return next_obs, reward, terminated, info

    def get_reward(self, actions, next_obs):
        self.reward_dict = {}
        if len(next_obs.shape) == 1:
            next_obs = np.expand_dims(next_obs, axis=0)
            actions = np.expand_dims(actions, axis=0)
            batch_mode = False
        else:
            batch_mode = True

        next_obs = np.array(next_obs)

        ee_pose = next_obs[:, -6:-3]  # Extract end-effector position from observation
        desired_pose = next_obs[:, -12:-9]  # Extract target position from observation

        qpos = next_obs[:, :self.model.nq]
        qvel = next_obs[:, self.model.nq:self.model.nq + self.model.nv]

        # 1. Trajectory cost
        traj_cost = np.linalg.norm(ee_pose - desired_pose, axis=1)
        
        # 2. Velocity penalty
        vel_penalty = np.sum(np.square(qvel), axis=1) 

        # Using negative cost as the reward directly.
        self.reward_dict["r_total"] = -(traj_cost + self.cv * vel_penalty)

        # return
        if not batch_mode:
            return self.reward_dict["r_total"][0]
        return self.reward_dict["r_total"].reshape(-1, 1)
    
    def is_terminated(self, act, next_obs):
        return False

    def reset_model(self):
        # Reset simulation
        if hasattr(self, 'model') and hasattr(self, 'data'):
            mujoco.mj_resetData(self.model, self.data)
        
        # Clear action buffer and populate with zeros
        self.action_buffer.clear()
        action_dim = self.model.nu if hasattr(self, 'model') else 7 # Default fallback
        for _ in range(self.delay_steps + 1):
            self.action_buffer.append(np.zeros(action_dim))
        
        self.prev_action = np.zeros(action_dim)
            
        obs = self._get_obs()
        return obs

    def reset(self, seed=None, options=None):
        # MujocoEnv.reset handles seed and calls reset_model
        obs, info = super().reset(seed=seed, options=options)
        
        return obs

    def render(self):
        ren = super().render()
        if ren is not None:
            if isinstance(self.frames, np.ndarray) and self.frames.size == 0:
                self.frames = []
            self.frames.append(ren)
        return ren
