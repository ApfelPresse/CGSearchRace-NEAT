import os

import neat

from feed_forward_converter import FeedForwardNetwork
from main import eval_genomes
from simulation import simulate

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-feedforward')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    population = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-31')
    winner = population.run(eval_genomes, 1)
    net = FeedForwardNetwork.create(winner, config)
    simulate(net, create_gif=True)
