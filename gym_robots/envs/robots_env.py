import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy
from random import randint

MAP_HEIGHT = 250
MAP_WIDTH = 250
AGENT_SIZE = 25

class RobotsEnv(gym.Env):
	def __init__(self, n_agents=2, max_steps=20):
		self.__map_shape = (MAP_WIDTH, MAP_HEIGHT)
		self.n_agents = n_agents
		self.max_steps = max_steps
		self.__count_steps = 0

		# TODO will fix this, currently we have an env with n agents where each can see x featrues describing the environment; 
		# thus atate = n * x and obs shape = x
		# this part doesnt need to be here, should be on the main file when running the env
		self.state_shape = 2 * 2 
		self.observation_shape = 2


		# TODO probably change this later; these are the bounds that the observation space values can assume
		self._obs_high = np.array([1., 1.])		
		self._obs_low = np.array([0., 0.])


		self.observation_space = [spaces.Box(low=self._obs_low, high=self._obs_high) for _ in range(self.n_agents)]
		self.action_space = [spaces.Discrete(5) for _ in range(self.n_agents)]

		self.agents_coords = list() # used to store the coordinates for the agents
		self.agents_front = list()  # to check the front side of the agents, if needed
		self.agent_dones = list()

		self.__map = None

		self.agents_prev_coords = list()



	def _init_env(self):
		self.agents_coords = list()
		self.agents_front = list()

		for agent_id in range(self.n_agents):
			start_x, start_y = randint(0, 225), randint(0, 225)  # 250 - 25 for now
			self.agents_coords.append([start_x, start_y])  # might change to float later
			self.agents_front.append(1)
		self.__count_steps = 0 

		#self.__build_map()

		print("Env initialized")


	def __build_map(self):
		n_rows = self.__map_shape[0]
		n_columns = self.__map_shape[1]

		self.__map = [[-1] * n_columns for _ in range(n_rows)]

		for agent_id in range(self.n_agents):
			self.__map[self.agents_coords[agent_id][0]][self.agents_coords[agent_id][1]] = agent_id

		#print(self.__map)

	def __update_map(self):
		for agent_id in range(self.n_agents):
			self.__map[self.agents_prev_coords[agent_id][0]][self.agents_prev_coords[agent_id][1]] = -1
			self.__map[self.agents_coords[agent_id][0]][self.agents_coords[agent_id][1]] = agent_id

		#print(self.__map)


	def __check_clash(self):
		# agents collide if the distance between corresponding vertexes is less than the width of each unit
		# considering for now each agent is represented by a square
		from math import sqrt

		coords = self.agents_coords

		for i in range(self.n_agents):
			for j in range(i+1, self.n_agents):
				dist_1 = sqrt((coords[j][0] - coords[i][0]) ** 2 + (coords[j][1] - coords[i][1]) ** 2)
				dist_2 = sqrt(((coords[j][0] + AGENT_SIZE) - (coords[i][0] + AGENT_SIZE)) ** 2 + (coords[j][1] - coords[i][1]) ** 2)
				dist_3 = sqrt((coords[j][0] - coords[i][0]) ** 2 + ((coords[j][1] + AGENT_SIZE) - (coords[i][1] + AGENT_SIZE)) ** 2)
				dist_4 = sqrt(((coords[j][0] + AGENT_SIZE) - (coords[i][0] + AGENT_SIZE)) ** 2 + ((coords[j][1] + AGENT_SIZE) - (coords[i][1] + AGENT_SIZE)) ** 2)

		if dist_1 <= AGENT_SIZE or dist_2 <= AGENT_SIZE or dist_3 <= AGENT_SIZE or dist_4 <= AGENT_SIZE:
			return True

		return False

	def _take_action(self, action, agent_id):
		self.agents_prev_coords = copy.deepcopy(self.agents_coords)
		action = ACTIONS[action]

		if action == 'FORWARD':
			if self.agents_coords[agent_id][0] + 2 * AGENT_SIZE <= self.__map_shape[0]:
				self.agents_coords[agent_id][0] += AGENT_SIZE  # fwd
		elif action == 'BACKWARD':
			if self.agents_coords[agent_id][0] - 2 * AGENT_SIZE >= 0:
				self.agents_coords[agent_id][0] -= AGENT_SIZE  # bwd
		elif action == 'UP':
			if self.agents_coords[agent_id][1] - 2 * AGENT_SIZE >= 0:
				self.agents_coords[agent_id][1] -= AGENT_SIZE  # left
		elif action == 'DOWN':
			if self.agents_coords[agent_id][1] + 2 * AGENT_SIZE <= self.__map_shape[1]:
				self.agents_coords[agent_id][1] += AGENT_SIZE  # right
			

		#print(self.__check_clash())
		# if action == noop

		#self.__update_map()

	def _check_done(self):
		if self.__count_steps == self.max_steps:
			self.agent_dones = [True for _ in range(self.n_agents)]
		return self.agent_dones


	# this should give full state info, which is an array with the n arrays with the partial observations of n agents
	def get_agent_obs(self):
		return self.agents_coords


	def step(self, action):
		obs = [0.]
		reward = [0, 0]
		episode_terminated = [False, False]
		
		for agent in range(self.n_agents):
			self._take_action(action[agent], agent)


		self.__count_steps += 1
		episode_terminated = self._check_done()

		if self.__check_clash():
			reward = list(map(lambda r : r - 5, reward))


		return obs, reward, episode_terminated, {}


	def render(self, mode='human'):
		from gym_robots.utils.rendering import Rendering
		renderer = Rendering(self.n_agents)

		renderer.draw(self.agents_coords)

	
	def reset(self):
		self._init_env()
		self.agent_dones = [False for _ in range(self.n_agents)]


	#def render(self):


	#def close(self):




# might change this to make it more realistic to roaming robots
ACTIONS = {
	0: 'NOOP',
	1: 'FORWARD',
	2: 'BACKWARD',
	3: 'UP',
	4: 'DOWN'
}
