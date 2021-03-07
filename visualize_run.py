import math
import os

import imageio
import matplotlib.pyplot as plt
import neat
import numpy as np

from feed_forward_converter import FeedForwardNetwork
from main import eval_genomes
from simulation import simulate


def convert_to_gif(name: str, frames: list):
    imageio.mimsave(f'./{name}.gif', frames, fps=5)


def plot_line(x, y, angle, length):
    # find the end point
    endy = y + length * math.sin(angle)
    endx = x + length * math.cos(angle)

    return [x, endx], [y, endy]


def plot_current_frame(checkpoints, current, car):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlim((0, 16000))
    ax.set_ylim((0, 9000))

    for j, checkpoint in enumerate(checkpoints):
        color = "b" if j == current else "r"
        ax.add_patch(plt.Circle((checkpoint.x, checkpoint.y), 400, color=color))

    ax.add_patch(plt.Circle((car.x, car.y), 300, color='g'))

    endx, endy = plot_line(car.x, car.y, car.angle, 1000)
    ax.plot(endx, endy)

    plt.axis('off')
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()
    return image


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-feedforward')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    population = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-99')
    winner = population.run(eval_genomes, 1)
    net = FeedForwardNetwork.create(winner, config)
    simulate(net, create_gif=True)
