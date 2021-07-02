from gym.envs.registration import register

N_AGENTS = 2

# just two agents roaming around
register(id='robots-v0',
		entry_point='gym_robots.envs:RobotsEnv',
		kwargs={'n_agents': N_AGENTS})

# two agents have to go to goal places and there are obstacles in the middle
register(id='robots-v1',
		entry_point='gym_robots.envs:RobotsEnv2',
		kwargs={'n_agents': N_AGENTS})

# to try the new idea
register(id='robots-v2',
		entry_point='gym_robots.envs:RobotsEnv3',
		kwargs={'n_agents': 1})