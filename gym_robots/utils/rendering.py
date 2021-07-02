import pygame
import os

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
BROWN = (210, 105, 30)

class Rendering():
	def __init__(self, n_agents, board=None, height=250, width=250, agent_size=25):
		self.window = pygame.display.set_mode((width, height))
		self.n_agents = n_agents 
		self.board = board
		self.height = height
		self.width = width
		self.agent_size = agent_size


	def draw(self, agents_coords):
		self.window.fill(BLACK)

		for agent_id in range(self.n_agents):
			x = agents_coords[agent_id][0]
			y = agents_coords[agent_id][1]

			pygame.draw.rect(self.window, WHITE, (x, y, self.agent_size, self.agent_size))

		pygame.display.update()


	def draw_v2(self, agents_coords, save=False, frame_counter=None, episode_i=None):
		'''for this version, we are using a 7*6 matrix, so we use here values on a scale 1:30 to
		make it compatible to the render screen
		- in this function, the map comes as a transposed matrix, since pygame window works with x,y coordinates
		and our map on the background is a 7x6 [row,column] matrix
		- the rendering is shown in the shape (row, column)

		'''
		actual_height = int(self.height / 30)
		actual_width = int(self.width / 30)

		self.window.fill(BLACK)

		# draw objective places and obstacles
		for i in range(actual_width):
			for j in range(actual_height):
				if self.board[i][j] == 999:
					if j <= actual_height - int(self.agent_size / 30):
						pygame.draw.rect(self.window, RED, (i * 30, j * 30, self.agent_size, self.agent_size))
					else:
						pygame.draw.rect(self.window, RED, ((i - self.agent_size) * 30, (j - self.agent_size) * 30, self.agent_size, self.agent_size))
				if self.board[i][j] == 998:
					pygame.draw.rect(self.window, BROWN, (i * 30, j * 30, self.agent_size, self.agent_size))
				


		# draw agents
		for agent_id in range(self.n_agents):
			x = agents_coords[agent_id][1] * 30
			y = agents_coords[agent_id][0] * 30

			if agent_id % 2 == 0:
				pygame.draw.rect(self.window, WHITE, (x, y, self.agent_size, self.agent_size))
			else:
				pygame.draw.rect(self.window, GREEN, (x, y, self.agent_size, self.agent_size))

		if not save:
			pygame.display.update()
		else:
			save_dir = "replays/episode_{}".format(episode_i)
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			pygame.image.save(self.window, save_dir + "/frame_{}.jpeg".format(frame_counter))



	def close(self):
		pygame.quit()