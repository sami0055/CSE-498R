import neat
from snake_game import SnakeGame
from model_utils import save_genome


def eval_genomes(genomes, config):
    """
    Evaluate each genome’s fitness by running a SnakeGame simulation.
    Fitness increases with food eaten and survival time.
    """
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = SnakeGame(width=20, height=20)
        steps = 0
        max_steps = 200  # limit to avoid infinite loops

        # Play game until done or step limit
        while True:
            state = game.get_state()
            output = net.activate(state)            # network output
            # 0=straight, 1=right, 2=left
            action = output.index(max(output))
            done, ate = game.step(action)
            steps += 1
            # Reward for surviving each step
            genome.fitness += 0.1
            # Reward for eating food
            if ate:
                genome.fitness += 10.0
            if done or steps > max_steps:
                break


def run_training(config_file):
    """
    Set up the NEAT population and run the evolution.
    """
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_file)
    pop = neat.Population(config)
    # Add reporters to show progress in the terminal
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    # Run evolution for N generations
    winner = pop.run(eval_genomes, 50)
    print("\nBest genome:\n", winner)
    # Save the best genome
    save_genome(winner, 'best_genome.pkl')


if __name__ == '__main__':
    run_training('config-feedforward.txt')
