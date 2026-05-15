import numpy as np
from torch.utils.tensorboard import SummaryWriter
import omegaconf
from tqdm import tqdm

import torch
from gym_env import FrankaEnv
from trajectories import CircleTrajectory


import mbrl
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util


seed = 0
traj = CircleTrajectory(center=[0.5, 0.0, 0.5], radius=0.1, angular_velocity=1.0)
env = FrankaEnv(traj, render_mode="rgb_array")
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape

reward_fn = env.get_reward
term_fn = env.is_terminated

device = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(seed=0)
generator = torch.Generator(device=device)
generator.manual_seed(seed)

trial_length = 200
num_trials = 10
ensemble_size = 5

# Everything with "???" indicates an option with a missing value.
# Our utility functions will fill in these details using the 
# environment information
cfg_dict = {
    # dynamics model configuration
    "dynamics_model": {
        "model": {
            "_target_": "mbrl.models.GaussianMLP",
            "device": device,
            "num_layers": 3,
            "ensemble_size": ensemble_size,
            "hid_size": 200,
            "in_size": "???",
            "out_size": "???",
            "deterministic": False,
            "propagation_method": "fixed_model",
            # can also configure activation function for GaussianMLP
            "activation_fn_cfg": {
                "_target_": "torch.nn.LeakyReLU",
                "negative_slope": 0.01
            }
        }
    },
    # options for training the dynamics model
    "algorithm": {
        "learned_rewards": False,
        "target_is_delta": True,
        "normalize": True,
    },
    # these are experiment specific options
    "overrides": {
        "trial_length": trial_length,
        "num_steps": num_trials * trial_length,
        "model_batch_size": 32,
        "validation_ratio": 0.05
    }
}
cfg = omegaconf.OmegaConf.create(cfg_dict)

# Create a 1-D dynamics model for this environment
dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

# Create a gym-like environment to encapsulate the model
model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn, generator=generator)

replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)

# Collect initial data with a random policy
common_util.rollout_agent_trajectories(
    env,
    trial_length, # initial exploration steps
    planning.RandomAgent(env),
    {}, # keyword arguments to pass to agent.act()
    replay_buffer=replay_buffer,
    trial_length=trial_length
)

print("# samples stored", replay_buffer.num_stored)

# CEM oplicy
agent_cfg = omegaconf.OmegaConf.create({
    # this class evaluates many trajectories and picks the best one
    "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
    "planning_horizon": 15,
    "replan_freq": 1,
    "verbose": False,
    "action_lb": "???",
    "action_ub": "???",
    # this is the optimizer to generate and choose a trajectory
    "optimizer_cfg": {
        "_target_": "mbrl.planning.CEMOptimizer",
        "num_iterations": 5,
        "elite_ratio": 0.1,
        "population_size": 500,
        "alpha": 0.1,
        "device": device,
        "lower_bound": "???",
        "upper_bound": "???",
        "return_mean_elites": True,
    }
})

agent = planning.create_trajectory_optim_agent_for_model(
    model_env,
    agent_cfg,
    num_particles=20
)

writer = SummaryWriter()

def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
    writer.add_scalar('Loss/training_loss', tr_loss, _total_calls)
    writer.add_scalar('Return/eval_return', val_score.mean().item(), _total_calls)

# Create a trainer for the model
model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

# Main PETS loop
for trial in range(num_trials):
    obs = env.reset(None)    
    agent.reset()
    
    terminated = False
    total_reward = 0.0
    print(f"======= Iteration {trial + 1} / {num_trials} =======")
    steps_trial = 0
    pbar = tqdm(total=trial_length, desc=f"Trial {trial + 1}", ncols=100)
    while not terminated:
        # --------------- Model Training -----------------
        if steps_trial == 0:
            dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats
            
            dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                replay_buffer,
                batch_size=cfg.overrides.model_batch_size,
                val_ratio=cfg.overrides.validation_ratio,
                ensemble_size=ensemble_size,
                shuffle_each_epoch=True,
                bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
            )
                
            model_trainer.train(
                dataset_train, 
                dataset_val=dataset_val, 
                num_epochs=50, 
                patience=50, 
                callback=train_callback)
            
        render_fn = env.render # if trial % 3 == 0 else None

        def render_callback(info):
            return render_fn() if render_fn is not None else None

        # --- Doing env step using the agent and adding to model dataset ---
        next_obs, reward, terminated, _ = common_util.step_env_and_add_to_buffer(
            env, obs, agent, {}, replay_buffer, callback=render_callback
        )
            
        obs = next_obs
        total_reward += reward
        steps_trial += 1
        pbar.update(1)
        if steps_trial == trial_length:
            break
    
    writer.add_scalar('Return/train_return', total_reward, trial)

    if hasattr(env, 'frames') and len(env.frames) > 0:
        # Save the frames as a video in TensorBoard
        video = np.stack(env.frames, axis=0)  # Shape: (num_frames, height, width, channels)
        video = np.expand_dims(video.transpose(0, 3, 1, 2), axis=0)  # Shape: (batch, num_frames, channels, height, width)
        writer.add_video(f'Trial_{trial + 1}_video', video, fps=10)
        env.frames = []


pbar.close()
writer.close()