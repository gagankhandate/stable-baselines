#!/usr/bin/env python3
import numpy as np
import gym

from stable_baselines.common.cmd_util import robotics_arg_parser, make_robotics_env
from stable_baselines import bench, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv


def train(env_id, num_timesteps, seed):
    """
    Train PPO2 model for Mujoco environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    """
    def make_env():
        env_out = make_robotics_env(env_id, seed)
        return env_out

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    model = PPO2(policy=policy, env=env, n_steps=16, nminibatches=8, lam=0.95, gamma=0.99, noptepochs=10,
                 ent_coef=0.0, learning_rate=3e-4, cliprange=0.2)
    model.learn(total_timesteps=num_timesteps)

    return model, env


def main():
    """
    Runs the test
    """
    args = robotics_arg_parser().parse_args()
    logger.configure()
    model, env = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

    logger.log("Running trained model")
    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    while True:
        actions = model.step(obs)[0]
        obs[:] = env.step(actions)[0]
        env.render()

if __name__ == '__main__':
    main()
