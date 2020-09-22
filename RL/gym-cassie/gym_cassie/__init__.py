from gym.envs.registration import register

register(
    id='cassie-v0',
    entry_point='gym_cassie.envs:CassieEnv',
)
