import socket
from struct import unpack

def go():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("127.0.0.1", 9000))
            cnt = 0
            while True:

                    data = s.recv(4) # speed
                    print(int(unpack(b'@f', data)[0] * 3.6))
                    
                    data = s.recv(4) # distance
                    #print(unpack(b'@f', data)[0])
                    
                    data = s.recv(4) # x
                    #print(unpack(b'@f', data)[0])
                    
                    data = s.recv(4) # y
                    #print(unpack(b'@f', data)[0])

                    data = s.recv(4) # z
                    #print(unpack(b'@f', data)[0])

                    data = s.recv(4) # steer
                    #print(unpack(b'@f', data)[0])

                    data = s.recv(4) # gas
                    #print(unpack(b'@f', data)[0])

                    data = s.recv(4) # brake
                    #print(unpack(b'@f', data)[0])

                    data = s.recv(4) # finish
                    #print(unpack(b'@f', data)[0])

                    data = s.recv(4) # gear
                    #print(unpack(b'@f', data)[0])

                    data = s.recv(4) # rpm
                    #print(unpack(b'@f', data)[0])



go()