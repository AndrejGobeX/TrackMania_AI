import socket
from struct import unpack
import threading

def get_data(s):
        data = dict()
        data['speed'] = unpack(b'@f', s.recv(4))[0] * 3.6 # speed
        data['distance'] = unpack(b'@f', s.recv(4))[0] # distance
        data['x'] = unpack(b'@f', s.recv(4))[0] # x
        data['y'] = unpack(b'@f', s.recv(4))[0] # y
        data['z'] = unpack(b'@f', s.recv(4))[0] # z
        data['steer'] = unpack(b'@f', s.recv(4))[0] # steer
        data['gas'] = unpack(b'@f', s.recv(4))[0] # gas
        data['brake'] = unpack(b'@f', s.recv(4))[0] # brake
        data['finish'] = unpack(b'@f', s.recv(4))[0] # finish
        data['gear'] = unpack(b'@f', s.recv(4))[0] # gear
        data['rpm'] = unpack(b'@f', s.recv(4))[0] # rpm
        return data
        
        