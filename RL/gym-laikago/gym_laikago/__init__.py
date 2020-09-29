from gym.envs.registration import register

register(
    id='laikago-v0',
    entry_point='gym_laikago.envs:LaikagoEnv',
)
