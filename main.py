import os
from multiprocessing import Pool

import neat

import visualize
from simulation import simulate


def run_fitness(configuration):
    genome_id, genome, config = configuration
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = simulate(net)
    return genome_id, fitness


def eval_genomes(genomes, config):
    configuration = []
    for genome_id, genome in genomes:
        configuration.append([genome_id, genome, config])
    with Pool(10) as pool:
        out = dict(pool.map(run_fitness, configuration))
        for genome_id, genome in genomes:
            genome.fitness = out[genome_id]


def plot_graphics(config, stats, winner):
    node_names = {
        -12: 'VY',
        -11: 'VX',
        -10: 'Angle',
        -9: 'Distance Check 1',
        -8: 'Car Y',
        -7: 'Car X',
        -4: 'Check Y 2',
        -3: 'Check X 2',
        -2: 'Check Y 1',
        -1: 'Check X 1',
        0: 'x',
        1: 'y',
        2: 'thrust'
    }
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-feedforward')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # p = neat.Population(config)
    p = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-99')

    # if True:
    #     population = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-99')
    #     winner = population.run(eval_genomes, 1)
    #     net = FeedForwardNetwork.create(winner, config)
    #     simulate(net, True)
    #     exit(0)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, filename_prefix="checkpoints/neat-checkpoint-"))

    winner = p.run(eval_genomes, 1000)

    print('\nBest genome:\n{!s}'.format(winner))

    plot_graphics(config, stats, winner)
