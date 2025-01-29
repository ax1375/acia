import gym
import numpy as np
from gym.envs.registration import register


def make(
    domain_name,
    task_name,
    noise,
    mult_factor,
    idx,
    seed=1,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None
):
    env_id = 'dmc_%s_%s_factor%s-v1' % (domain_name, task_name, idx)

    if from_pixels:
        assert not visualize_reward, 'cannot use visualize reward when learning from pixels'

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if not env_id in gym.envs.registry.env_specs:
        register(
            id=env_id,
            entry_point='dmc2gym.wrappers:DMCWrapper',
            kwargs={
                'domain_name': domain_name,
                'task_name': task_name,
                'noise': noise,
                'mult_factor': mult_factor,
                'idx': idx,
                'task_kwargs': {
                    'random': seed
                },
                'environment_kwargs': environment_kwargs,
                'visualize_reward': visualize_reward,
                'from_pixels': from_pixels,
                'height': height,
                'width': width,
                'camera_id': camera_id,
                'frame_skip': frame_skip,
            },
            max_episode_steps=max_episode_steps
        )
    return gym.make(env_id)
