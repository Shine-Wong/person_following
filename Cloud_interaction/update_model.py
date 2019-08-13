import socket
import os
import threading
import time


# server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 9999))
s.listen(1)
print("waiting for connection")


def file_scp(sock, addr):
    print("accept message from {}".format(addr))
    while True:
        data = sock.recv(1024)
        if not data:
            break
        current_time = time.strftime('%m%d%H')
        a = os.system('scp -r saved_networks model_backup/%s' % current_time)
        if a == 0:
            print("has backuped current model successfully")
        b = os.system(
            'scp -r plz@219.216.87.170:~/D_mysql/saved_networks ./')
        if b == 0:
            print("has updated the newest local model")


while True:
    sock, addr = s.accept()
    t = threading.Thread(target=file_scp, args=(sock, addr))
    t.start()
