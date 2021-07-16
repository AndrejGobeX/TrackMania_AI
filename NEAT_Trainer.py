# Training script for NEAT
#
# Switch to Trackmania and press s to begin
# The script will simulate all runs automatically for the number of generations
#
# python NEAT_Trainer PID address [endian]

from __future__ import print_function
import os
import neat
import keyboard
import SpeedCapture
import DirectKey
import sys
import time
import numpy as np
from PIL import ImageGrab, ImageEnhance, Image
import cv2
# In order to visualize the training net, you need to copy visualize.py file into the NEAT directory (you can find it in the NEAT repo)
# Because of the licence, I am not going to copy it to my github repository
# You can still train your network without it
try:
    import NEAT.visualize as visualize
except ModuleNotFoundError:
    print('Missing visualize.py file.')

os.chdir('./NEAT')
print(os.getcwd())

if len(sys.argv) < 3:
    print('Not enough arguments.')
    exit()

# trackmania PID, speed address and endian

PID = int(sys.argv[1])
address = sys.argv[2]
if address[:2] != '0x':
    address = '0x' + address
address = int(address, 0)

if len(sys.argv) < 4:
    endian = 'little'
else:
    endian = sys.argv[4]

# image dimensions
image_width = 100
image_height = 100

# hyperparams
threshold = 0.5
no_generations = 20
max_fitness = 100.0
no_seconds = 20
kill_seconds = 5
kill_speed = 51 / 300.0
no_lines = 5
checkpoint = 'neat-checkpoint-4'

up = False
down = False
left = False
right = False

KEY_UP = 0xC8
KEY_DOWN = 0xD0
KEY_LEFT = 0xCB
KEY_RIGHT = 0xCD
KEY_DELETE = 0xD3

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def initial_crop(img, l, u, r, d):
    img = img.crop((l, u, img.size[0]-r, img.size[1]-d))
    return img


def mod_edge(img, w, h):
    img = initial_crop(img, 0, img.size[1]//2, 0, img.size[1]//3)
    img = ImageEnhance.Contrast(img).enhance(2).convert('L')  # .filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = img.resize((w, h), Image.ANTIALIAS)
    img = np.array(img)
    img = cv2.medianBlur(img, 5)
    img = (img < 50) * np.uint8(255)
    return img.reshape((h, w, 1))


def mod_shrink_n_measure(img, w, h, no_lines):
    img_np = mod_edge(img, w, h)
    return find_walls(img_np, no_lines=no_lines)


def find_walls(img_np, no_lines=10, threshold=200):
    h, w, d = img_np.shape
    dx = w//no_lines

    end_points = []

    start_points = range(dx//2, w, dx)
    for start_point in start_points:
        distance = h - 1
        while distance >= 0:
            if img_np[distance][start_point] >= threshold:  # pixel threshold
                break
            distance -= 1
        distance = h - distance - 1
        end_points.append(distance * 1.0 / h)
    
    return end_points


def run_inference(img_np, end_points):
    no_lines = len(end_points)
    h, w, d = img_np.shape
    dx = w//no_lines

    if d == 1:
        img_np = np.stack((img_np,)*3, axis=-1).reshape(h, w, 3)

    start_points = range(dx//2, w, dx)
    for start_point, end_point in zip(start_points, end_points):
        distance = end_point * h
        while distance > 0:
            i = int(h - distance)
            if i >= h:
                i = h - 1
            img_np[i][start_point][0] = 0
            img_np[i][start_point][1] = 255
            img_np[i][start_point][2] = 255
            distance -= 1
    
    return img_np


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 10.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Driving
        print("Ready. Genome id: " + str(genome_id))

        '''while not keyboard.is_pressed('q'):
            if not keyboard.is_pressed('s'):
                continue'''

        begin = time.time()
        y = []
        x = []
        DirectKey.PressKey(KEY_DELETE)
        DirectKey.ReleaseKey(KEY_DELETE)

        while True:

            # uncomment for reaction time measurement
            # start = time.time()

            # screenshot
            img = ImageGrab.grab()

            img = mod_shrink_n_measure(img, image_width, image_height, no_lines)

            try:
                img = img / 255.0
            except:
                img = img

            # speed
            speed = SpeedCapture.GetSpeed(PID, address, endian=endian) / 300.0

            img.append(speed)
            y.append(speed)
            x.append(time.time()-begin)

            # net
            output = np.array(net.activate(img))
            output = output > threshold

            up = output[2]
            #down = output[2]
            left = output[1]
            right = output[0]

            if up:
                DirectKey.PressKey(KEY_UP)
            else:
                DirectKey.ReleaseKey(KEY_UP)

            if down:
                DirectKey.PressKey(KEY_DOWN)
            else:
                DirectKey.ReleaseKey(KEY_DOWN)

            if left:
                DirectKey.PressKey(KEY_LEFT)
            else:
                DirectKey.ReleaseKey(KEY_LEFT)

            if right:
                DirectKey.PressKey(KEY_RIGHT)
            else:
                DirectKey.ReleaseKey(KEY_RIGHT)

            # stop = time.time()
            # print(stop - start) # reaction time
            finish = time.time()-begin
            if finish > no_seconds or (finish > kill_seconds and speed < kill_speed):#keyboard.is_pressed('f'):
                DirectKey.ReleaseKey(KEY_UP)
                DirectKey.ReleaseKey(KEY_DOWN)
                DirectKey.ReleaseKey(KEY_LEFT)
                DirectKey.ReleaseKey(KEY_RIGHT)
                break
        
        #print('Time:')
        #t = float(input())
        genome.fitness = np.trapz(y, x)#max_fitness - t




def run(config_file, checkpoint=None):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    if not checkpoint == None:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to global no generations.
    winner = p.run(eval_genomes, no_generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    node_names = {0:'r', 1:'l', 2:'u'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-9')
    # p.run(eval_genomes, 10)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
print('Press s to begin.')
keyboard.wait('s')

if checkpoint == None:
    for cpt in os.listdir('.'):
        if cpt[:4] == 'neat':
            os.unlink('./'+cpt)

run(config_path, checkpoint)
