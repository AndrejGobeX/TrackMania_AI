import socket
from struct import unpack
import threading
from time import sleep

def get_data(s):
        data = dict()
        data['speed'] = unpack(b'@f', s.recv(4))[0] # speed
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
        data['dx'] = unpack(b'@f', s.recv(4))[0] # dx
        data['dy'] = unpack(b'@f', s.recv(4))[0] # dy
        data['dz'] = unpack(b'@f', s.recv(4))[0] # dz
        return data
        

if __name__ == "__main__":

        data = {}

        # function that captures data from openplanet    
        def data_getter_function():
                global data
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect(("127.0.0.1", 9000))
                        while True:
                                data = get_data(s)

        # start the thread
        data_getter_thread = threading.Thread(target=data_getter_function, daemon=True)
        data_getter_thread.start()

        sleep(0.2) # wait for connection

        racing_line = []

        while not data['finish']:
                racing_line.append([data['x'], data['z'], data['dx'], data['dz']])
                sleep(0.1)
        print(racing_line)