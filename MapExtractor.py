from matplotlib import colors
import matplotlib.pyplot as plt
import sys
import numpy as np

xr = []
yr = []

xl = []
yl = []

blocks = []

flag = ''
next = -1

with open(sys.argv[1], 'r') as map:
    for line in map:
        line = line[:-1]
        if line == "RoadTechStart":
            flag = 'g'
        elif line == "RoadTechFinish":
            flag = 'r'
        elif flag == 'g' or flag == 'r':
            xyz = [int(e) for e in line[1:-1].split(",")]
            plt.scatter([xyz[0]+16], [xyz[2]+16], color=flag)
            flag = ''
        elif line == '-':
            next += 1
            blocks.append([[],[],[],[]])
        else:
            rxylxy = [float(e.replace(",", ".")) for e in line.split(" ")]
            for i in range(4):
                blocks[next][i].append(rxylxy[i])

for block in blocks:
    plt.plot(block[0], block[1], color = 'tab:blue')
    plt.plot(block[2], block[3], color = 'tab:orange')
plt.gca().invert_xaxis()
plt.axis("equal")
plt.show()