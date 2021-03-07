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


def get_pool():
    return Pool(4)


def eval_genomes(genomes, config):
    configuration = []
    for genome_id, genome in genomes:
        configuration.append([genome_id, genome, config])
    pool = get_pool()
    out = dict(pool.map(run_fitness, configuration))
    for genome_id, genome in genomes:
        genome.fitness = out[genome_id]


def plot_graphics(config, stats, winner):
    node_names = {
        -9: 'VY',
        -8: 'VX',
        -7: 'Angle',
        -6: 'Distance Check 2',
        -5: 'Distance Check 1',
        -4: 'Car Y',
        -3: 'Car X',
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

    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-949')

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50, filename_prefix="checkpoints/neat-checkpoint-"))

    pool = Pool(8)
    get_pool = lambda: pool

    try:
        winner = p.run(eval_genomes, 10000)
    finally:
        pool.close()

    print('\nBest genome:\n{!s}'.format(winner))

    plot_graphics(config, stats, winner)
