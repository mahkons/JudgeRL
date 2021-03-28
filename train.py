import pybullet_envs
# Don't forget to install PyBullet!

import gym
import numpy as np
import torch
from torch import nn
import random

from PPO import PPO
from params import  ENV_NAME, MIN_EPISODES_PER_UPDATE, MIN_TRANSITIONS_PER_UPDATE, ITERATIONS


def evaluate_policy(env, agent, episodes):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns
   

def sample_episode(env, agent, judge):
    s = env.reset()
    done = False
    judge_trajectory = list()
    trajectory = list()
    while not done:
        a, pa, p = agent.act(s)
        ja, jpa, jp = judge.act(np.concatenate([s, a]))

        v = agent.get_value(s)
        ns, r, done, _ = env.step(a)

        judge_trajectory.append((np.concatenate([s, a]), jpa, r, jp, v))
        trajectory.append((s, pa, ja, p, v))

        s = ns
    return trajectory, judge_trajectory


def train():
    env = gym.make(ENV_NAME)

    judge = PPO(state_dim=env.observation_space.shape[0] + env.action_space.shape[0], action_dim=1)
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])

    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0

    best = 0
    
    for i in range(ITERATIONS):
        trajectories = list()
        judge_trajectories = list()
        steps_ctn = 0
        
        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj, jtraj = sample_episode(env, ppo, judge)
            steps_ctn += len(traj)
            trajectories.append(traj)
            judge_trajectories.append(jtraj)

        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)        
        judge.update(judge_trajectories)

        s = sum([t[2] for tt in trajectories for t in tt])
        print("lol {}".format(s))
        
        if (i + 1) % (ITERATIONS//100) == 0:
            rewards = evaluate_policy(env, ppo, 20)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            val = np.mean(rewards)
            if val > best:
                best = val
                ppo.save()


def init_random_seeds(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


if __name__ == "__main__":
    init_random_seeds(23)
    train()
