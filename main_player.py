import pygame
import numpy as np
import socket, pickle
from reversi import reversi
from reversi_model import ReversiModel
import time 



def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()
    reversi_model = ReversiModel()

    while True:
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return
        game.board = board
       
        predicted_move = reversi_model.predict(board)
        
        #MiniMax Algorithm  - Replace with your algorithm
        #x,y = minimax.minimax_Algorithm(game, turn, depth)
        # Print the chosen move
        print("Selected move:", (predicted_move[0], predicted_move[1])) #x, y))
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