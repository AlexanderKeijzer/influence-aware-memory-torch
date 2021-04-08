import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import VecEnvWrapper

from model import IAMBase


def main():

    seed = 1
    env_name = "Warehouse-v0"
    num_processes = 32
    log_dir ='./logs/'
    eval_interval = None
    log_interval = 10
    use_linear_lr_decay = False
    use_proper_time_limits = False
    save_dir = './trained_models/'
    use_cuda = True

    # PPO
    gamma = 0.99 # reward discount factor
    clip_param = 0.1#0.2
    ppo_epoch = 3#4
    num_mini_batch = 32
    value_loss_coef = 1#0.5
    entropy_coef = 0.01
    lr = 2.5e-4#7e-4
    eps = 1e-5
    max_grad_norm = float('inf')
    use_gae = True
    gae_lambda = 0.95
    num_steps = 8#5

    # Store
    num_env_steps = 10e6
    save_interval = 100

    # IAM
    dset = [0, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
        63, 64, 65, 66, 67, 68, 69, 70, 71, 72]


    #gym.envs.register(env_name, entry_point="environments.warehouse.warehouse:Warehouse",
    #                    kwargs={'seed': seed, 'parameters': {"num_frames": 1}})

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    log_dir = os.path.expanduser(log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    #envs = Monitor(gym.make(env_name), os.path.join(log_dir, str(0)),
    #                      allow_early_resets=False)

    envs = make_vec_envs(env_name, seed, num_processes,
                        gamma, log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base=IAMBase,
        base_kwargs={'dset': dset})
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=lr,
        eps=eps,
        max_grad_norm=max_grad_norm)

    rollouts = RolloutStorage(num_steps, num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        num_env_steps) // num_steps // num_processes
    for j in range(num_updates):

        if use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, lr)

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        

        rollouts.compute_returns(next_value, use_gae, gamma,
                                 gae_lambda, use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % save_interval == 0
                or j == num_updates - 1) and save_dir != "":
            save_path = os.path.join(save_dir, 'PPO')
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, env_name + ".pt"))

        if j % log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * num_processes * num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (eval_interval is not None and len(episode_rewards) > 1
                and j % eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, env_name, seed,
                     num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()