import socket
import struct
import threading
import time


class GameDataGetter:
    # GameData info indexes
    I_SPEED = 0
    I_DISTANCE = 1
    I_X = 2
    I_Y = 3
    I_Z = 4
    I_STEER = 5
    I_GAS = 6
    I_BRAKE = 7
    I_FINISH = 8
    I_GEAR = 9
    I_RPM = 10
    I_DX = 11
    I_DY = 12
    I_DZ = 13
    # GameData array length
    SIZE = 11

    def __init__(self, localhost = "127.0.0.1", port = 9000, extended=False):
        if extended:
            GameDataGetter.SIZE += 3
        self.game_data = [0.0]*GameDataGetter.SIZE
        self.localhost = localhost
        self.port = port

        # start the thread
        self.thread = threading.Thread(target=self._thread_function, daemon=True)
        self.thread.start()

        # wait for connection
        time.sleep(0.2)

    def _fetch_data(self, s):
        i = 0
        while i < GameDataGetter.SIZE:
            self.game_data[i] = struct.unpack(b'@f', s.recv(4))[0]
            i += 1

    def _thread_function(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.localhost, self.port))
            while True:
                self._fetch_data(s)

if __name__ == "__main__":
    data_getter = GameDataGetter(extended=True)
    while True:
        print(data_getter.game_data[GameDataGetter.I_DX])
