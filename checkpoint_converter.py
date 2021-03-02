import os

import neat

from feed_forward_converter import convert_net_to_base64
from main import eval_genomes


def checkpoint_to_base64(checkpoint: str, config):
    population = neat.Checkpointer.restore_checkpoint(checkpoint)
    winner = population.run(eval_genomes, 1)
    convert_net_to_base64(winner, config)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-feedforward')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    checkpoint_to_base64("checkpoints/neat-checkpoint-0", config)
