import pygame
import neat
from snake_game import SnakeGame
from model_utils import load_genome

# Load the best genome and NEAT config
config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    'config-feedforward.txt')
genome = load_genome('best_genome.pkl')
net = neat.nn.FeedForwardNetwork.create(genome, config)

# Initialize Pygame
pygame.init()
CELL_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = 20, 20
screen = pygame.display.set_mode((GRID_WIDTH*CELL_SIZE, GRID_HEIGHT*CELL_SIZE))
pygame.display.set_caption("Snake NEAT Replay")
clock = pygame.time.Clock()

game = SnakeGame(width=GRID_WIDTH, height=GRID_HEIGHT)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Network decides action
    state = game.get_state()
    output = net.activate(state)
    action = output.index(max(output))
    done, ate = game.step(action)

    # Draw background
    screen.fill((0, 0, 0))
    # Draw snake
    for x, y in game.snake:
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(
            x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    # Draw food
    fx, fy = game.food
    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(
        fx*CELL_SIZE, fy*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.display.flip()

    clock.tick(10)  # control game speed

    if done:
        pygame.time.wait(500)
        running = False

pygame.quit()
