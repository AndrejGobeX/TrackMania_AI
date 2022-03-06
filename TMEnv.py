import gym
from gym import spaces
from time import sleep, time
import numpy as np
import socket
import threading

from MapExtractor import get_map_data, plot_map
from Commands import tm_accelerate, tm_brake, tm_reset, tm_respawn, tm_steer, tm_update
import GetData

STEER = 0
ACCELERATE = 1
BRAKE = 2
RND = 1

TIME = 20.0

CHECKPOINT_REWARD = 100.0
WALL_REWARD = -7.0

def check_checkpoint(p, r, q, s):
  rxs = np.cross(r, s)
  qmp = q - p
  qpxs = np.cross(qmp, s)
  qpxr = np.cross(qmp, r)
  if rxs == 0:
    return np.array([-1, -1])
  t = qpxs/rxs
  u = qpxr/rxs
  if t >= 0 and t <= 1 and u >= 0 and u <= 1:
    return p + t*r
  return np.array([-1, -1])


def normalize_speed(speed):
  return speed * 0.0036


class TMEnv(gym.Env):
  """Trackmania Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, map_path, verbose=False):
    super(TMEnv, self).__init__()
    self.action_space = spaces.Box(
        np.array([-1, 0, 0], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32) # steer, gas, brake
    )
    self.observation_space = spaces.Box(
        np.array(np.zeros(14), dtype=np.float32), np.array(np.ones(14), dtype=np.float32) # data
    )

    self.map_path = map_path
    self.racing_line_path = map_path.split('.txt')[0] + '.racing.line'
    self.done = False
    self.verbose = verbose
    self.timer = time()
    self.stopwatch = 0.0
    self.previous_stopwatch = 0.0

    # function that captures data from openplanet    
    def data_getter_function():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("127.0.0.1", 9000))
            while True:
                self.data = GetData.get_data(s)

    # start the thread
    self.data_getter_thread = threading.Thread(target=data_getter_function, daemon=True)
    self.data_getter_thread.start()

    sleep(0.2) # wait for connection

    self.previous_location = np.round([self.data['x'], self.data['z']], RND)
    self.location = np.round([self.data['x'], self.data['z']], RND)

    # get blocks of the map
    self.blocks = get_map_data(map_path)
    self.current_block_index = 0 # start at the first block
    self.current_checkpoint_index = 0
    self.no_blocks = len(self.blocks)
    # mini checkpoints, first is skipped since it is the last of the previous block, start block's first is never crossed
    self.racing_line = np.array([
        np.round([np.append((np.array([block[0][i], block[1][i]])+np.array([block[2][i], block[3][i]]))/2, 100) for i in range(1, len(block[0]))], RND) # x, y, time
    for block in self.blocks], dtype=object)

    self.direction = self.racing_line[0][0][:-1] - np.array([(self.blocks[0][0][0]+self.blocks[0][2][1])/2, (self.blocks[0][1][0]+self.blocks[0][3][1])/2])
    #self.previous_difference = self.racing_line[0][0][:-1] - self.location
    #self.previous_difference = np.linalg.norm(self.previous_difference)


  def save_racing_line(self):
    with open(self.racing_line_path, 'wb') as file:
      np.save(file, self.racing_line)

  def load_racing_line(self):
    with open(self.racing_line_path, 'rb') as file:
      self.racing_line = np.load(file, allow_pickle=True)

  def move(self):
    reward = 0.0
    intersection = [0, 0]
    while intersection[0] != -1:
      cp_r = np.array([self.blocks[self.current_block_index][0][self.current_checkpoint_index+1], self.blocks[self.current_block_index][1][self.current_checkpoint_index+1]])
      cp_l = np.array([self.blocks[self.current_block_index][2][self.current_checkpoint_index+1], self.blocks[self.current_block_index][3][self.current_checkpoint_index+1]])
      intersection = check_checkpoint(
        self.location,
        self.previous_location - self.location,
        cp_r,
        cp_l - cp_r
      )
      if intersection[0] != -1:
        small_reward = self.racing_line[self.current_block_index][self.current_checkpoint_index][2] - self.stopwatch
        #if small_reward > 0: TODO: uncomment
          # update racing line
          #self.racing_line[self.current_block_index][self.current_checkpoint_index] = np.append(intersection, self.stopwatch)
        if self.verbose:
          print("Crossed mini checkpoint "+str(self.current_block_index)+", "+str(self.current_checkpoint_index))
          print([intersection, self.stopwatch])
        self.current_checkpoint_index += 1
        # if final in block
        if self.current_checkpoint_index == len(self.racing_line[self.current_block_index]):
          self.current_block_index += 1
          self.current_checkpoint_index = 0
        reward += CHECKPOINT_REWARD # + small_reward TODO: uncomment

    #difference = self.racing_line[self.current_block_index][self.current_checkpoint_index][:-1] - self.location
    #difference = np.linalg.norm(difference)
    #if reward == 0:
    #  reward += (self.previous_difference - difference)/36*20
    #self.previous_difference = difference

    return reward

  def next_checkpoint(self):
    dy = self.direction[1]
    dx = self.direction[0]
    θ = 0.0
    if dx == 0:
      θ = np.pi/2
      θ *= np.sign(dy)
    else:
      k = dy/dx
      θ = np.arctan(k)
      if dx < 0.0:
        θ += np.pi
    
    checks = 3
    bi = self.current_block_index
    ci = self.current_checkpoint_index-1
    obs = []
    while checks > 0:
      checks -= 1
      ci += 1
      if self.no_blocks <= bi:
        obs.append(1)
        obs.append(0)
        continue
      if len(self.racing_line[bi]) <= ci:
        ci = 0
        bi += 1
        if self.no_blocks <= bi:
          obs.append(1)
          obs.append(0)
          continue
      delta_point = self.racing_line[bi][ci][:-1] - self.location
      x = delta_point[0]*np.cos(θ) + delta_point[1]*np.sin(θ)
      y = -delta_point[0]*np.sin(θ) + delta_point[1]*np.cos(θ)
      ret = np.clip([x, y], -100.0, 100.0)/100
      ret[0] = np.abs(ret[0])*2-1
      obs.append(ret[0])
      obs.append(ret[1])

    if self.flip:
      for i in range(len(obs)):
        if i&1:
          obs[i] = -obs[i]
    return np.array(obs)

  def measure(self, theta):
    blocks = self.blocks
    block_i = self.current_block_index
    location = self.location
    prev_location = self.previous_location

    vx = np.sign(self.direction[0])
    vy = np.sign(self.direction[1])
    edge_case = False
    if vx == 0:
        angle = np.pi/2*vy
    else:
        k = (self.direction[1])/(self.direction[0])
        angle = np.arctan(k)
        if vx < 0:
            angle += np.pi
    angle += theta
    if np.abs(angle) == np.pi/2:
        edge_case = True
    else:
        k = np.tan(angle)
        n = location[1] - k*location[0]
    
    vx = np.sign(np.round(np.cos(angle), RND))
    vy = np.sign(np.round(np.sin(angle), RND))
    start_block = block_i - 2 if block_i > 2 else 0
    finish_block = block_i + 3 if block_i + 3 < len(blocks) else len(blocks)
    shortest_distance = 100
    best_intersection = [0,0]
    for i in range(start_block, finish_block):
        block = blocks[i]
        for j in range(len(block[0])-1):
            if (block[0][j+1]-block[0][j]) == 0:
                if edge_case:
                    continue
                intersection = [block[0][j+1], block[0][j+1]*k+n]
            else:
                k1 = (block[1][j+1]-block[1][j])/(block[0][j+1]-block[0][j])
                n1 = block[1][j+1] - k1*block[0][j+1]
                if edge_case:
                    intersection = [location[0], k1*location[0]+n1]
                else:
                    intersection = [(n1-n)/(k-k1),0]
                    intersection[1] = k*intersection[0] + n
            dx = np.sign(intersection[0] - block[0][j]) * np.sign(intersection[0] - block[0][j+1])
            dy = np.sign(intersection[1] - block[1][j]) * np.sign(intersection[1] - block[1][j+1])
            if dx <= 0 and dy <= 0:
                vx1 = np.sign(intersection[0] - location[0])
                vy1 = np.sign(intersection[1] - location[1])
                if vx == vx1 and vy == vy1:
                    distance = np.sqrt((intersection[0]-location[0])**2 + (intersection[1]-location[1])**2)
                    if distance < shortest_distance:
                        shortest_distance = distance
                        best_intersection = intersection
        for j in range(len(block[0])-1):
            if (block[2][j+1]-block[2][j]) == 0:
                if edge_case:
                    continue
                intersection = [block[2][j+1], block[2][j+1]*k+n]
            else:
                k1 = (block[3][j+1]-block[3][j])/(block[2][j+1]-block[2][j])
                n1 = block[3][j+1] - k1*block[2][j+1]
                if edge_case:
                    intersection = [location[0], k1*location[0]+n1]
                else:
                    intersection = [(n1-n)/(k-k1),0]
                    intersection[1] = k*intersection[0] + n
            dx = np.sign(intersection[0] - block[2][j]) * np.sign(intersection[0] - block[2][j+1])
            dy = np.sign(intersection[1] - block[3][j]) * np.sign(intersection[1] - block[3][j+1])
            if dx <= 0 and dy <= 0:
                vx1 = np.sign(intersection[0] - location[0])
                vy1 = np.sign(intersection[1] - location[1])
                if vx == vx1 and vy == vy1:
                    distance = np.sqrt((intersection[0]-location[0])**2 + (intersection[1]-location[1])**2)
                    if distance < shortest_distance:
                        shortest_distance = distance
                        best_intersection = intersection
    return [best_intersection, shortest_distance]


  def step(self, action):
    self.previous_stopwatch = self.stopwatch
    self.stopwatch = time() - self.timer
    self.done = bool(self.data['finish'])
    reward = (self.previous_stopwatch - self.stopwatch)*100
    #if self.done:
    if self.stopwatch > TIME: # drive for 15 sec
      self.done = True
    
    new_location = np.round([self.data['x'], self.data['z']], RND)
    if np.abs(new_location-self.location).sum() >= 1:
      self.previous_location = self.location
      self.location = new_location
      self.direction = self.location - self.previous_location
    print(np.round(np.arctan(self.direction[0]/self.direction[1])/np.pi*180, RND))
    tm_steer(action[STEER]*(-1 if self.flip else 1))
    tm_accelerate(action[ACCELERATE]/2+0.5)
    if self.data['speed'] < 7.0:
      tm_brake(0.0)
    else:
      tm_brake(action[BRAKE]/2+0.5)
    tm_update()
    reward += self.move()
    if self.flip:
      walls = [self.measure(i)[1]/100.0 for i in [-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2]]
    else:
      walls = [self.measure(i)[1]/100.0 for i in [np.pi/2, np.pi/3, np.pi/6, 0, -np.pi/6, -np.pi/3, -np.pi/2]]
    for wall in walls:
      reward += (wall < 0.2)*WALL_REWARD
    walls.append(normalize_speed(self.data['speed']))
    if walls[-1] < 0.05 and walls[3] < 0.1:
      reward -= 10000
      self.done = True
    return np.append(self.next_checkpoint(), walls), reward, self.done, {}

  def reset(self):
    self.done = False
    self.flip = (np.random.rand()*2 > 1)
    self.current_block_index = 0
    self.current_checkpoint_index = 0
    tm_reset()
    tm_update()
    tm_respawn()
    sleep(1.5)
    self.direction = self.racing_line[0][0][:-1] - np.array([(self.blocks[0][0][0]+self.blocks[0][2][1])/2, (self.blocks[0][1][0]+self.blocks[0][3][1])/2])
    self.location = np.round([self.data['x'], self.data['z']], RND)
    self.previous_location = np.round([self.data['x'], self.data['z']], RND)
    if self.flip:
      walls = [self.measure(i)[1]/100.0 for i in [-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2]]
    else:
      walls = [self.measure(i)[1]/100.0 for i in [np.pi/2, np.pi/3, np.pi/6, 0, -np.pi/6, -np.pi/3, -np.pi/2]]
    walls.append(normalize_speed(self.data['speed']))
    self.stopwatch = 0.0
    self.previous_stopwatch = 0.0
    self.timer = time()
    return np.append(self.next_checkpoint(), walls)

  def render(self, mode='human'):
    pass

  def close (self):
    pass