from matplotlib import colors
import matplotlib.pyplot as plt
import sys
import numpy as np


def get_map_data(file):
    flag = ''
    next = -1
    blocks = []
    with open(file, 'r') as map:
        for line in map:
            line = line[:-1]
            if line == '-':
                next += 1
                blocks.append([[],[],[],[]])
            else:
                rxylxy = [float(e.replace(",", ".")) for e in line.split(" ")]
                for i in range(4):
                    blocks[next][i].append(rxylxy[i])
    return blocks


def plot_map(blocks):
    for block in blocks:
        plt.plot(block[0], block[1], color = 'tab:blue')
        plt.plot(block[2], block[3], color = 'tab:orange')
    plt.gca().invert_xaxis()
    plt.axis("equal")


if __name__ == '__main__':
    blocks = get_map_data(sys.argv[1])
    plot_map(blocks)
    plt.show()