import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy
from random import randint
import random
from math import sqrt

'''building for a map that should look like the grid in the lab 7x6 matrix
.... so in this version, the coordinates are seen as (row,column); on
the other version they are seen as (x,y)
-- on this side it works as (r,c); to render on the other side, rows and columns are converted to (x,y) coords
but on this side its always (r,c) matrix line of thinking

-------------------------------------------------------------
- with the current config, if they reach the goals without either clashing with each other or crash into an object,
the maximum joint reward shoud be sum([5, 5]) = 10


TODO:
- fix observations
- create other class to use sample method for an array of actions with n agents

'''

######################################### NOTES ####################################################
# for now there is just one agent and one goal place and one obstacle
# to use a fixed obstacle comment the lines that place it randomly and uncommment the ones putting it in the middle of the map

MAP_HEIGHT = 7
MAP_WIDTH = 6
AGENT_SIZE = 1

class RobotsEnv3(gym.Env):
	# use obstacles is a flag to use the obstacles in the environment and account for collisions; remove goal is a flag to ignore the goal place and 
	# the objective is just to avoid the obstacles; static_obstacle makes the obstacle static in a random starting position
	# a non static object moves randomly alongside the middle row 
	def __init__(self, n_agents=2, max_steps=50, use_obstacles=True, remove_goal=False, static_obstacle=True):
		self.__map_shape = (MAP_HEIGHT, MAP_WIDTH)
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
		self.action_space = [spaces.Discrete(4) for _ in range(self.n_agents)]

		self.agents_coords = list() # used to store the coordinates for the agents
		self.agents_front = list()  # to check the front side of the agents, if needed
		self.agent_dones = list()
		self.goals_reached = list()  # agent reached the goal

		self.__map = None
		self.obstacles = []
		self.use_obstacles = use_obstacles
		self.goal = []# [[MAP_HEIGHT - AGENT_SIZE, MAP_WIDTH - AGENT_SIZE], [0, 0]] #[] # [[MAP_HEIGHT - AGENT_SIZE, MAP_WIDTH - AGENT_SIZE], [0, 0]]  # define goal places for both agents  # uncomment this for fixed goal places

		self.agents_prev_coords = list()  # stores previous coordinates to update map
		self.prov_coords = list()  # provisional array to backup the agents coordinates to check for collisions before updating map
		self.obstacles_prev_coords = list()  # stores previous obstacles coordinates to update map
		self.obstacles_prov_coords = list()  # provisional array to backup the obstacles coordinates to check for collisions before updating map

		self.frame_counter = 0
		self.remove_goal = remove_goal
		self.static_obstacle = static_obstacle



	def _init_env(self):
		self.agents_coords = list()
		self.agents_front = list()
		self.goals_reached = [False for agent_id in range(self.n_agents)]

		self.goal = [[randint(0, MAP_HEIGHT - AGENT_SIZE), randint(0, MAP_WIDTH - AGENT_SIZE)], [0, 0]] #[[MAP_HEIGHT - AGENT_SIZE, MAP_WIDTH - AGENT_SIZE], [0, 0]]  # TODO: for now just working with one agent

		start_r, start_c = randint(0, MAP_HEIGHT - AGENT_SIZE), randint(0, MAP_WIDTH - AGENT_SIZE) #0, 0

		for agent_id in range(self.n_agents):
			
			while self.goal[agent_id][0] == start_r and self.goal[agent_id][1] == start_c:
				start_r, start_c = randint(0, MAP_HEIGHT - AGENT_SIZE), randint(0, MAP_WIDTH - AGENT_SIZE)  # 250 - 25 for now
			
			self.agents_coords.append([start_r, start_c])  # might change to float later
			self.agents_front.append(1)
		self.__count_steps = 0 

		# initialize obstacles
		if self.use_obstacles:
			while True:
				flag = 0
				# for now just one obstacle
				# lets palce it in the middle of the map
				self.obstacles = [[int(MAP_HEIGHT / 2), randint(0, MAP_WIDTH - AGENT_SIZE)]] # [[randint(0, MAP_HEIGHT - AGENT_SIZE), randint(0, MAP_WIDTH - AGENT_SIZE)]]
				
				for agent_id in range(self.n_agents):
					if self.agents_coords[agent_id][0] == self.obstacles[0][0] and self.agents_coords[agent_id][1] == self.obstacles[0][1] \
					or self.goal[0][0] == self.obstacles[0][0] and self.goal[0][1] == self.obstacles[0][1]:
						flag = 1

				if flag == 0:
					break 

			
		self.__build_map()

		#print("Env initialized")


	def __print_map(self):
		for row in self.__map:
			print(row)


	def __build_map(self):
		n_rows = self.__map_shape[0]
		n_columns = self.__map_shape[1]

		self.__map = [[-1] * n_columns for _ in range(n_rows)]

		for agent_id in range(self.n_agents):
			self.__map[self.agents_coords[agent_id][0]][self.agents_coords[agent_id][1]] = agent_id

		#self.__map[0][0] = 999
		#self.__map[MAP_HEIGHT - AGENT_SIZE][MAP_WIDTH - AGENT_SIZE] = 999
		self.__map[self.goal[0][0]][self.goal[0][1]] = 999

		#if self.use_obstacles:
		#	self.obstacles = [[int(MAP_HEIGHT / 2), int(MAP_WIDTH / 2)]]

		#self.__print_map()

	def __check_goal(self):
		if self.remove_goal:
			return [False for _ in range(self.n_agents)]

		dones = []
		for agent_id in range(self.n_agents):
			dones.append(False)
			if self.agents_coords[agent_id][0] == self.goal[agent_id][0] and self.agents_coords[agent_id][1] == self.goal[agent_id][1]:
				dones[agent_id] = True
		return dones

	def __obstacle_colision(self):
		for i in range(len(self.obstacles)):
			for agent_id in range(self.n_agents):
				if self.obstacles[i][0] == self.prov_coords[agent_id][0] and self.obstacles[i][1] == self.prov_coords[agent_id][1]:
					return agent_id
		return -1


	def __update_map(self):
		for agent_id in range(self.n_agents):
			self.__map[self.agents_prev_coords[agent_id][0]][self.agents_prev_coords[agent_id][1]] = -1
			self.__map[self.agents_coords[agent_id][0]][self.agents_coords[agent_id][1]] = agent_id

		for agent_id in range(self.n_agents):
			self.__map[self.goal[agent_id][0]][self.goal[agent_id][1]] = 999

		for i in range(len(self.obstacles)):
			self.__map[self.obstacles_prev_coords[i][0]][self.obstacles_prev_coords[i][1]] = -1
			self.__map[self.obstacles[i][0]][self.obstacles[i][1]] = 998

		#print(self.__map)
		#self.__print_map()

	def __check_clash(self):
		# in this env the movements are discrete, so colision is simply if both step into the same cell
		# considering for now each agent is represented by a square

		for i in range(self.n_agents):
			for j in range(i+1, self.n_agents):
				if self.prov_coords[i][0] == self.prov_coords[j][0] and self.prov_coords[i][1] == self.prov_coords[j][1]:
					return True

		return False

	def _take_action(self, action, agent_id):
		action = ACTIONS[action]

		self.prov_coords = copy.deepcopy(self.agents_coords)

		dones = self.__check_goal()

		if not dones[agent_id]:
			if action == 'FORWARD':
				if self.agents_coords[agent_id][1] + 2 * AGENT_SIZE <= self.__map_shape[1]:
					self.prov_coords[agent_id][1] += AGENT_SIZE  # fwd
			elif action == 'BACKWARD':
				if self.agents_coords[agent_id][1] - AGENT_SIZE >= 0:
					self.prov_coords[agent_id][1] -= AGENT_SIZE  # bwd
			elif action == 'UP':
				if self.agents_coords[agent_id][0] - AGENT_SIZE >= 0:
					self.prov_coords[agent_id][0] -= AGENT_SIZE  # left
			elif action == 'DOWN':
				if self.agents_coords[agent_id][0] + 2 * AGENT_SIZE <= self.__map_shape[0]:
					self.prov_coords[agent_id][0] += AGENT_SIZE  # right
		
		# if agent_id collides with any other agent
		if self.__check_clash():
			return 'agent_crash'

		# if agent_id collides with an object
		if self.__obstacle_colision() != -1:
			#self.render()
			#f = input()
			return 'obstacle_crash'

		# if it didint collide, the action is valid and the agent can perform
		self.agents_coords = copy.deepcopy(self.prov_coords)
		return 'safe_action'

		#print(self.__check_clash())
		# if action == noop

		#self.__update_map()

	def _check_done(self):
		if self.__count_steps == self.max_steps:
			self.agent_dones = [True for _ in range(self.n_agents)]
			return self.agent_dones

		return self.agent_dones


	# this should give full state info, which is an array with the n arrays with the partial observations of n agents
	def get_agent_obs(self):
		obs = copy.deepcopy(self.agents_coords)
		
		for agent_id in range(self.n_agents):
			# normalize coordinates to lie between 0 and 1
			obs[agent_id][0] = round(obs[agent_id][0] / (self.__map_shape[0] - 1), 2)
			obs[agent_id][1] = round(obs[agent_id][1] / (self.__map_shape[1] - 1), 2)

			# observations regarding neighbor places 
			p1_r = self.agents_coords[agent_id][0] + 1
			p1_c = self.agents_coords[agent_id][1]

			# change obs[agent_id].append(0) to obs[agent_id].append(1) to include map borders in the mask
			if p1_r >= self.__map_shape[0]:
				obs[agent_id].append(0)
			elif self.__map[p1_r][p1_c] != -1 and self.__map[p1_r][p1_c] != 999:
				obs[agent_id].append(1)
			else:
				obs[agent_id].append(0)


			p2_r = self.agents_coords[agent_id][0] - 1
			p2_c = self.agents_coords[agent_id][1]

			if p2_r < 0:
				obs[agent_id].append(0)
			elif self.__map[p2_r][p2_c] != -1 and self.__map[p2_r][p2_c] != 999:
				obs[agent_id].append(1)
			else:
				obs[agent_id].append(0)


			p3_r = self.agents_coords[agent_id][0]
			p3_c = self.agents_coords[agent_id][1] + 1

			if p3_c >= self.__map_shape[1]:
				obs[agent_id].append(0)
			elif self.__map[p3_r][p3_c] != -1 and self.__map[p3_r][p3_c] != 999:
				obs[agent_id].append(1)
			else:
				obs[agent_id].append(0)


			p4_r = self.agents_coords[agent_id][0]
			p4_c = self.agents_coords[agent_id][1] - 1

			if p4_c < 0:
				obs[agent_id].append(0)
			elif self.__map[p4_r][p4_c] != -1 and self.__map[p4_r][p4_c] != 999:
				obs[agent_id].append(1)
			else:
				obs[agent_id].append(0)

			# compute distance to goal place to use in observation
			# norm agent_coords
			norm_agent_r = round(obs[agent_id][0] / (self.__map_shape[0] - 1), 2)
			norm_agent_c = round(obs[agent_id][1] / (self.__map_shape[1] - 1), 2)

			norm_goal_r = round(self.goal[0][0] / (self.__map_shape[0] - 1), 2)
			norm_goal_c = round(self.goal[0][1] / (self.__map_shape[1] - 1), 2)


			dist = sqrt((norm_goal_r - norm_agent_r) ** 2 + \
					(norm_goal_c - norm_agent_c) ** 2)

			dist = round(dist, 3)

			obs[agent_id].append(dist)
			obs[agent_id] = tuple(obs[agent_id])


		return obs


	def __move_obstacles(self):
		# just one obstacle for now# for now move alongside middle row
		move_vectors = [(0, 1), (0, -1)] # [(1, 0), (0, 1), (-1, 0), (0, -1)]  # row, columns
		obstacle_move = random.choice(move_vectors)

		self.obstacles_prov_coords = copy.deepcopy(self.obstacles)
		
		
		if obstacle_move == move_vectors[0]:
			if self.obstacles[0][1] + 2 * AGENT_SIZE <= self.__map_shape[1]:
				self.obstacles_prov_coords[0][1] += AGENT_SIZE  # fwd
		elif obstacle_move == move_vectors[1]:
			if self.obstacles[0][1] - AGENT_SIZE >= 0:
				self.obstacles_prov_coords[0][1] -= AGENT_SIZE  # bwd

		if (self.agents_coords[0][0] != self.obstacles_prov_coords[0][0] or self.agents_coords[0][1] != self.obstacles_prov_coords[0][1]) \
				and (self.goal[0][0] != self.obstacles_prov_coords[0][0] or self.goal[0][1] != self.obstacles_prov_coords[0][1]):
			self.obstacles = copy.deepcopy(self.obstacles_prov_coords)
	
		#if obstacle_move == move_vectors[1]:
		#	if self.obstacles[0][1] + 2 * AGENT_SIZE <= self.__map_shape[1]:
		#		self.obstacles[0][1] += AGENT_SIZE  # fwd
		#elif obstacle_move == move_vectors[3]:
		#	if self.obstacles[0][1] - AGENT_SIZE >= 0:
		#		self.obstacles[0][1] -= AGENT_SIZE  # bwd
		#elif obstacle_move == move_vectors[2]:
		#	if self.obstacles[0][0] - AGENT_SIZE >= 0:
		#		self.obstacles[0][0] -= AGENT_SIZE  # left
		#elif obstacle_move == move_vectors[0]:
		#	if self.obstacles[0][0] + 2 * AGENT_SIZE <= self.__map_shape[0]:
		#		self.obstacles[0][0] += AGENT_SIZE  # right




	def step(self, action):
		obs = [0.]
		reward = [0, 0]
		ob_crash = 0
		count_goal = 0
		episode_terminated = [False, False]

		self.obstacles_prev_coords = copy.deepcopy(self.obstacles)

		# move obstacles
		if not self.static_obstacle:
			self.__move_obstacles()
		
		self.agents_prev_coords = copy.deepcopy(self.agents_coords)
		
		for agent in range(self.n_agents):
			#reward[agent] += self.__count_steps * 0.01
			tag = self._take_action(action[agent], agent)
			if tag == 'agent_crash':  # agent_id crashed with other agent
				reward[agent] -= 5
			elif tag == 'obstacle_crash':  # agent_id collided with obstacle
				reward[agent] -= 2
				ob_crash = 1

		self.__update_map()


		goal_checkup = self.__check_goal()

		for i in range(self.n_agents):
			if not self.goals_reached[i] and goal_checkup[i]:
				reward[i] = 5
				count_goal += 1


		self.__count_steps += 1
		episode_terminated = self._check_done()

		self.goals_reached = self.__check_goal()
		
		if all(self.goals_reached):
			episode_terminated = [True for _ in range(self.n_agents)]
			#self.render()
			#f=input()

		'''if self.__check_clash():
			reward = list(map(lambda r : r - 5, reward))

		
		colision_id = self.__obstacle_colision()
		if colision_id != -1:
			reward[colision_id] -= 2'''


		return self.get_agent_obs(), reward, episode_terminated, {'crash':ob_crash, 'count_goal':count_goal}


	def render(self, mode='human'):
		from gym_robots.utils.rendering import Rendering
		renderer = Rendering(self.n_agents, np.transpose(self.__map), MAP_HEIGHT * 30, MAP_WIDTH * 30, AGENT_SIZE * 30)

		renderer.draw_v2(self.agents_coords)

	def save_replay(self, epsiode_i):
		from gym_robots.utils.rendering import Rendering
		renderer = Rendering(self.n_agents, np.transpose(self.__map), MAP_HEIGHT * 30, MAP_WIDTH * 30, AGENT_SIZE * 30)

		renderer.draw_v2(self.agents_coords, True, self.frame_counter, epsiode_i)
		self.frame_counter += 1


	
	def reset(self):
		self._init_env()
		self.frame_counter = 0
		self.agent_dones = [False for _ in range(self.n_agents)]

		return self.get_agent_obs()


	#def render(self):


	#def close(self):




# might change this to make it more realistic to roaming robots
#ACTIONS = {
#	0: 'NOOP',
#	1: 'FORWARD',
#	2: 'BACKWARD',
#	3: 'UP',
#	4: 'DOWN'
#}

ACTIONS = {
	0: 'FORWARD',
	1: 'BACKWARD',
	2: 'UP',
	3: 'DOWN'
}

