from matplotlib import colors
import matplotlib.pyplot as plt
import sys
import numpy as np


def get_map_data_blocks(file):
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


def get_map_data(file):
    blocks = [[0, 0, 0, 0]]
    with open(file, 'r') as map:
        for line in map:
            line = line[:-1]
            if line == '-':
                blocks.pop()
            else:
                rxylxy = [float(e.replace(",", ".")) for e in line.split(" ")]
                blocks.append(rxylxy[0:4])
    return np.array(blocks)


def plot_map(blocks):
    plt.plot(blocks.T[0], blocks.T[1], color = 'tab:blue')
    plt.plot(blocks.T[2], blocks.T[3], color = 'tab:orange')
    plt.gca().invert_xaxis()
    plt.axis("equal")


def plot_map_blocks(blocks):
    for block in blocks:
        plt.plot(block[0], block[1], color = 'tab:blue')
        plt.plot(block[2], block[3], color = 'tab:orange')
    plt.gca().invert_xaxis()
    plt.axis("equal")


if __name__ == '__main__':
    blocks = get_map_data(sys.argv[1])
    plot_map(blocks)
    plt.show()