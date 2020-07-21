# Claudio Perez
# June 2020

###########################################################
# # To find wsl host ip: 
# from subprocess import run, PIPE
# import re
# wsl2_ip = re.search(r'\sinet ((?:\d+\.){3}\d+)/', run(
#     'wsl -e ip -4 addr show dev eth0'.split(),
#     stdout=PIPE, encoding='utf8', check=True).stdout)[1]
# HOST = wsl2_ip
###########################################################

from params import *
import socket, json

PORT = 9999
SIZE = 64
FORMAT = 'utf-8'
FAM = socket.AF_INET

HOST = '172.24.79.108'

ADDR = (HOST, PORT)
# print('Client: Addr: {}'.format(ADDR))

with socket.socket(FAM, socket.SOCK_STREAM) as client:
    client.connect(ADDR)
    # print('Client: Connection successfull.')
    param = json.dumps([E, P, Ao, Au]).encode(FORMAT)

    client.send(param)

    result = json.loads(client.recv(SIZE).decode(FORMAT))
    client.close()
    # print('Client: Message received: {}'.format(result))

with open('results.out', 'w') as f:
    f.write('{:.60g} {:.60g}'.format(result[1],result[0]))

