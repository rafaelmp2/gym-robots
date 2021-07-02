import pyglet
from pyglet.gl import *
import pygame
import sys
import random

W_HEIGHT = 250
W_WIDTH = 250
CELL_SIZE = 10
CELL_COLOR = (255, 255, 255)

MAP_DIM = (10, 10)
PLAYERS_COORDS = [[2,4], [7,8]]
MAP = [
	[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
	[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
	[-1,-1,-1,-1, 0,-1,-1,-1,-1,-1],
	[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
	[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
	[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
	[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
	[-1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
	[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
	[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
]

window = pygame.display.set_mode((W_WIDTH, W_HEIGHT))
'''
for i in range(0, int(W_WIDTH/CELL_SIZE)):
	for j in range(0, int(W_HEIGHT/CELL_SIZE)):
		pygame.draw.rect(window, CELL_COLOR, (0 + CELL_SIZE * i, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))
'''
'''
for player in PLAYERS_COORDS:
	for x,y in PLAYERS_COORDS:
		pygame.draw.rect(window, CELL_COLOR, (x, y, CELL_SIZE, CELL_SIZE))

'''


while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_LEFT:
				window.fill((0,0,0))
				rand_x, rand_y = random.randint(0, 100), random.randint(0, 100)
				pygame.draw.rect(window, CELL_COLOR, (rand_x, rand_y, CELL_SIZE, CELL_SIZE))


			
	pygame.display.update()