#Zijie Zhang, Sep.24/2023

import pygame
import numpy as np
import socket, pickle
from reversi import reversi
from minimax_algorithmPTH import MiniMax
import time 

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()
    minimax = MiniMax()
    
    #############
    max_depth = 4 #figure out correct depth or if needs to be adjusted dynamically
    ###############

    while True:

        #Receive play request from the server
        #turn : 1 --> you are playing as white | -1 --> you are playing as black
        #board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)
        start = time.time()
        #Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return
        game.board = board
        #Debug info
        print(turn)
        depth = 2
        while time.time() - start < 5:
            if depth <= max_depth:
                #MiniMax Algorithm  - Replace with your algorithm
                x,y = minimax.minimax_Algorithm(game, turn, depth)
                depth += 1
            else:
                break
        
        #MiniMax Algorithm  - Replace with your algorithm
        x,y = minimax.minimax_Algorithm(game, turn, depth)
        # Print the chosen move
        print("Selected move:", (x, y))
        print()
        
        ####
        #  So turn doesn't need to be updated since it does so on its own as long as x, y is -1, -1
        # look at greedy player as an example of how it runs. Your now in the minimax algorithm.
        # we still the entire contents of game since thats whats being used to step and read board
        # could turn not be updated?
        ##########
        game.step(x,y, turn, True)
        
        
        
        
        ###############################
        
        #Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x,y]))
        
if __name__ == '__main__':
    main()