import math
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
FLOAT_PRECISION = 3

TIME = 10000000.0
TIME_START = 3.0
WALL_CONTACT_FRONT = 5.75/100.0
WALL_CONTACT_WHEELS = 5.75/100.0
WALL_CONTACT_SIDE = 4.5/100.0
WALL_COEF = 800
WALL_PENALTY = 1
GAMMA_ADJUSTMENT = -math.log(0.01)/60.0

# util functions
def cross_product(x, y):
  return x[0]*y[1] - x[1]*y[0]


def norm(x):
    return math.sqrt(x[0]**2 + x[1]**2)


def vector_intersection(p, r, q, s):
  rxs = cross_product(r, s)
  qmp = q - p
  qpxs = cross_product(qmp, s)
  qpxr = cross_product(qmp, r)
  if rxs == 0:
    return np.array([-1, -1])
  t = qpxs/rxs
  u = qpxr/rxs
  if t >= 0 and t <= 1 and u >= 0 and u <= 1:
    return p + t*r
  return np.array([-1, -1])


def vector_angle(r):
    n = norm(r)
    if n == 0.0:
      return 0.0
    v = r[0]/n
    θ = np.arccos(v)
    if r[1] < 0:
      θ *= -1
    return θ


def three_point_circle_inverse_radius(A, B, C):
  B -= A
  C -= A
  B2 = B[0]**2 + B[1]**2
  C2 = C[0]**2 + C[1]**2
  #D=2[Ax(By−Cy)+Bx(Cy−Ay)+Cx(Ay−By)]
  D=2*(B[0]*C[1] - C[0]*B[1])
  if D == 0.0:
    return 0.0
  #Rx=[(Ax**2+Ay**2)(By−Cy)+(Bx**2+By**2)(Cy−Ay)+(Cx**2+Cy**2)(Ay−By)]/D
  Rx=(B2*C[1] - C2*B[1])/D
  #Ry=[(Ax**2+Ay**2)(Cx−Bx)+(Bx**2+By**2)(Ax−Cx)+(Cx**2+Cy**2)(Bx−Ax)]/D
  Ry=(-B2*C[0] + C2*B[0])/D
  return 1/math.sqrt(Rx**2 + Ry**2)


def curvature(centerline, cur_block, cur_projection, delta=50, no_points=2):
    iter_block = cur_block
    battery = delta
    points = []
    while no_points>0:
        no_points-=1
        while battery > 0:
            if iter_block == len(centerline):
                break
            
            dif = centerline[iter_block] - cur_projection
            z = norm(dif)
            battery -= z
            
            if battery < 0:
                dif*=battery/z
                cur_projection = centerline[iter_block]+dif
                break

            cur_projection = centerline[iter_block]
            iter_block += 1
        battery = delta
        points.append(cur_projection)
    return points


'''def sensors(blocks, cur_block, xy, angle, θ=0, max_length=100, start=4, finish=4):
    # angle
    #angle = np.arccos(dxdy[0]) * np.sign(dxdy[1]) + θ
    angle += θ
    v = np.array([np.cos(angle)*max_length, np.sin(angle)*max_length])
    
    # block range
    start_block = cur_block - start
    start_block = start_block if start_block > -1 else 0
    finish_block = cur_block + finish
    finish_block = finish_block if finish_block <= len(blocks) else len(blocks)

    # search
    best_distance = max_length
    no_point = np.array([-1, -1])
    best_point = no_point

    for i in range(start_block, finish_block):
        block = blocks[i]
        for j in range(len(block[0])-1):
            for s in range(0, 3, 2):
                wall_first = np.array([block[s][j], block[s+1][j]])
                wall_second = np.array([block[s][j+1], block[s+1][j+1]])
                intersection = vector_intersection(xy, v, wall_first, wall_second - wall_first)
                if not np.array_equal(intersection, no_point):
                    d = norm(xy-intersection)
                    if d < best_distance:
                        best_distance = d
                        best_point = intersection
    return [best_point, best_distance]'''


def sensors(blocks, cur_block, xy, angle, θ=0, max_length=100, start=10, finish=30):
    # angle
    #angle = np.arccos(dxdy[0]) * np.sign(dxdy[1]) + θ
    angle += θ
    v = np.array([np.cos(angle)*max_length, np.sin(angle)*max_length])
    
    # block range
    start_block = cur_block - start
    start_block = start_block if start_block > -1 else 0
    finish_block = cur_block + finish
    finish_block = finish_block if finish_block < len(blocks) else len(blocks)-1

    # search
    best_distance = max_length
    no_point = np.array([-1, -1])
    best_point = no_point

    for i in range(start_block, finish_block):
        for s in range(0, 3, 2):
            wall_first = np.array([blocks[i][s], blocks[i][s+1]])
            wall_second = np.array([blocks[i+1][s], blocks[i+1][s+1]])
            intersection = vector_intersection(xy, v, wall_first, wall_second - wall_first)
            if not np.array_equal(intersection, no_point):
                d = norm(xy-intersection)
                if d < best_distance:
                    best_distance = d
                    best_point = intersection
                else:
                  break
    return [best_point, best_distance]


def normalize_speed(speed):
  return speed * 0.0036

# gym class
class TMEnv(gym.Env):
  """Trackmania Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, map_path):
    super(TMEnv, self).__init__()

    # action and observation
    self.action_space = spaces.Box(
        # steer, gas/brake
        np.array([-1, -1], dtype=np.float32), np.array([1, 1], dtype=np.float32)
    )
    self.observation_space = spaces.Box(
        # 7 x walls, speed, θ, previous steer, wall contact, curvature
        np.array(np.array([0]*8 + [-np.pi, -1, 0, 0]), dtype=np.float32), np.array(np.array([1]*8 + [np.pi, 1, 1, 1]), dtype=np.float32) # data
    )

    # env setup
    self.map_path = map_path
    self.no_point = np.array([-1, -1])
    self.flip = True
    self.previous_step = time()

    # get blocks of the map
    self.blocks = get_map_data(map_path)
    self.centerline = [
      (np.array([self.blocks[i][0], self.blocks[i][1]]) + np.array([self.blocks[i][2], self.blocks[i][3]]))/2
      for i in range(len(self.blocks))] # x, y
    self.track_length = 0
    for i in range(len(self.centerline)-2):
      self.track_length += norm(self.centerline[i]-self.centerline[i+1])
    print("Track centerline length: "+str(self.track_length))

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

    self.speed = 0.0

    self.rspwn = True
    self.respawn()


  def respawn(self):
    ''' reset the car and respawn '''
    tm_reset()
    tm_update()
    if self.rspwn:
      tm_respawn()
      sleep(1.8)

    self.timer = time()
    self.stopwatch = 0.0

    self.update()

    if self.rspwn:
      self.next_checkpoint = 1
      self.previous_projection = self.location.copy()
      self.rspwn = False
      self.previous_steer = 0

    self.previous_steer = -self.previous_steer  
    self.flip = not self.flip
    print("Flipped: ", self.flip)


  def update(self):
    ''' update the state '''
    self.stopwatch = time() - self.timer
    reward = 0.0
    self.done = bool(self.data['finish'])
    if self.done: reward = 1000.0
    if self.stopwatch > TIME:
      self.done = True
    self.location = np.array([round(self.data['x'], FLOAT_PRECISION), round(self.data['z'], FLOAT_PRECISION)])
    self.direction = np.array([round(self.data['dx'], FLOAT_PRECISION), round(self.data['dz'], FLOAT_PRECISION)])
    self.angle = vector_angle(self.direction)
    self.previous_speed = self.speed
    self.speed = normalize_speed(self.data['speed'])
    return reward


  def move(self):
    ''' calculate reward '''
    reward = self.update() #- 1.0
    θ = 0.0
    contact = 0.0
    
    # next block
    v = -self.direction*100
    w = self.blocks[self.next_checkpoint][0:2] - self.blocks[self.next_checkpoint][2:4]
    while not np.array_equal(vector_intersection(self.location, v, self.blocks[self.next_checkpoint][2:4], w), self.no_point):
            reward += norm(self.centerline[self.next_checkpoint] - self.previous_projection)
            self.previous_projection = self.centerline[self.next_checkpoint]
            self.next_checkpoint += 1
            if self.next_checkpoint >= len(self.centerline):
              self.next_checkpoint = 1
              self.done = True
              self.rspwn = True
            w = self.blocks[self.next_checkpoint][0:2] - self.blocks[self.next_checkpoint][2:4]

    # projection on the centerline
    k = self.centerline[self.next_checkpoint][0:2]-self.centerline[self.next_checkpoint-1][0:2]
    if k[0] == 0:
        projection = np.array([self.centerline[self.next_checkpoint][0],self.location[1]])
    else:
        k = k[1]/k[0]
        n = self.centerline[self.next_checkpoint][1] - k*self.centerline[self.next_checkpoint][0]
        projection = (self.location[0] + k*self.location[1] - k*n)/(1+k**2)
        projection = np.array([
            projection,
            projection*k + n
        ])
    
    n1 = norm(self.centerline[self.next_checkpoint]-self.previous_projection)
    n2 = norm(self.centerline[self.next_checkpoint]-projection)
    # advancement
    if n2 < n1:
      reward += n1-n2
      self.previous_projection = projection

    # gamma
    reward *= math.e ** (-self.stopwatch*GAMMA_ADJUSTMENT)

    # angle between the centerline
    θ = vector_angle(self.centerline[self.next_checkpoint]-self.centerline[self.next_checkpoint-1]) - self.angle
    if θ > np.pi:
      θ -= 2*np.pi
    if self.flip:
      θ = -θ

    # curvature
    c = curvature(self.centerline, self.next_checkpoint, self.previous_projection)
    ir = three_point_circle_inverse_radius(self.location, c[0], c[1])

    # walls
    walls = []
    for i in [np.pi/2, np.pi/3, np.pi/6, 0, -np.pi/6, -np.pi/3, -np.pi/2]:
      if self.flip:
        i = -i
      walls.append(
        sensors(self.blocks, self.next_checkpoint, self.location, self.angle, i)[1]/100.0
      )
    if walls[3] < WALL_CONTACT_FRONT or \
      min(walls[0], walls[-1]) < WALL_CONTACT_SIDE or \
      min(walls[1], walls[2], walls[4], walls[5]) < WALL_CONTACT_WHEELS:

      contact = 1.0
      reward -= contact*self.previous_speed**2*WALL_COEF + WALL_PENALTY

    if self.speed < 0.005 and self.stopwatch > TIME_START:
      self.rspwn = self.done = True
    
    '''
    if self.next_checkpoint == 0:
      # fell out of bounds
      reward -= 10000
      self.done = True
    '''
    
    # 7 x walls, speed, θ, previous steer, wall contact, curvature
    # inputs, reward, done, {}
    return [np.array(walls + [self.speed, θ, self.previous_steer, contact, ir], dtype=np.float32), reward, self.done, {}]


  def step(self, action):
    print("Step diff(s): ", time() - self.previous_step)
    self.previous_step = time()
    tm_reset()
    self.previous_steer = action[0]
    if self.flip:
      action[0] = -action[0]
    tm_steer(action[0])
    if action[1] > 0.2:
      tm_accelerate(1)
    elif self.speed > 0.05:
      tm_brake(-action[1])
    tm_update()
    return tuple(self.move())


  def reset(self):
    self.previous_step = time()
    self.rspwn |= bool(self.data['finish'])
    self.respawn()
    self.update()

    return np.array(self.move()[0])


  def render(self, mode='human'):
    pass


  def close (self):
    pass