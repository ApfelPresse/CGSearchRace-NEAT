import imageio
import matplotlib.pyplot as plt
import numpy as np


def convert_to_gif(name: str, frames: list):
    imageio.mimsave(f'./{name}.gif', frames, fps=5)


def plot_current_frame(checkpoints, current, car):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlim((0, 16000))
    ax.set_ylim((0, 9000))

    for j, checkpoint in enumerate(checkpoints):
        color = "b" if j == current else "r"
        ax.add_patch(plt.Circle((checkpoint.x, checkpoint.y), 400, color=color))

    ax.add_patch(plt.Circle((car.x, car.y), 300, color='g'))

    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()
    return image
