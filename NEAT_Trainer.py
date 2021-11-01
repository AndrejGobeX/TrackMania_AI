# Training script for NEAT
#
# Switch to Trackmania and press s to begin
# The script will simulate all runs automatically for the number of generations
#
# python NEAT_Trainer

from __future__ import print_function
import os
import neat
import keyboard
import DirectKey
import sys
import time
import numpy as np
from PIL import ImageGrab, ImageEnhance, Image
import cv2
import socket
from struct import unpack
import vgamepad as vg
import GetData
import threading
# In order to visualize the training net, you need to copy visualize.py file into the NEAT directory (you can find it in the NEAT repo)
# Because of the licence, I am not going to copy it to my github repository
# You can still train your network without it
try:
    import NEAT.visualize as visualize
except ModuleNotFoundError:
    print('Missing visualize.py file.')

os.chdir('./NEAT')
print(os.getcwd())

if len(sys.argv) < 0:
    print('Not enough arguments.')
    exit()

# image dimensions
image_width = 100
image_height = 100

# hyperparams
#threshold = 0.5
no_generations = 20
#max_fitness = 100.0
no_seconds = 20
kill_seconds = 5
kill_speed = 51
no_lines = 5
checkpoint = None

up = False
down = False
left = False
right = False

KEY_UP = 0xC8
KEY_DOWN = 0xD0
KEY_LEFT = 0xCB
KEY_RIGHT = 0xCD
KEY_DELETE = 0xD3

gamepad = vg.VX360Gamepad()

def data_getter_function():
    global data
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("127.0.0.1", 9000))
        while True:
            data = GetData.get_data(s)


data_getter_thread = threading.Thread(target=data_getter_function, daemon=True)
data_getter_thread.start()

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

        distance = 0.0
        time.sleep(1)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as SOCKET:
            SOCKET.connect(("127.0.0.1", 9000))

            DirectKey.PressKey(KEY_DELETE)
            DirectKey.ReleaseKey(KEY_DELETE)
            begin = time.time()


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
                speed = data['speed']
                distance = data['distance']

                img.append(speed)

                # net
                output = np.array(net.activate(img))

                brake = 0.0#output[2]
                gas = output[1]
                steer = output[0]

                steer = steer * 2 - 1
                
                gamepad.left_joystick_float(x_value_float=steer, y_value_float=0)
                gamepad.right_trigger_float(value_float=gas)
                gamepad.left_trigger_float(value_float=brake)
                gamepad.update()

                # stop = time.time()
                # print(stop - start) # reaction time
                finish = time.time()-begin
                if finish > no_seconds: #or (finish > kill_seconds and speed < kill_speed):
                    gamepad.reset()
                    gamepad.update()
                    break
            
            genome.fitness = distance




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
    try:
        visualize.draw_net(config, winner, True, node_names=node_names)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
    except:
        print('Missing visualize.py file.')

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-9')
    # p.run(eval_genomes, 10)

local_dir = os.getcwd()
config_path = os.path.join(local_dir, 'config-feedforward')
print('Press s to begin.')
keyboard.wait('s')

if checkpoint == None:
    for cpt in os.listdir('.'):
        if cpt[:4] == 'neat':
            os.unlink('./'+cpt)

run(config_path, checkpoint)
