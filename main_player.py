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
        
        # Print the chosen move
        print("Selected move:", (predicted_move[0], predicted_move[1])) #x, y))
        print()
        x = predicted_move[0]
        y = predicted_move[1]
        
        game.step(x,y, turn, True)
        
        ###############################
        
        #Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x,y]))
        
if __name__ == '__main__':
    main()