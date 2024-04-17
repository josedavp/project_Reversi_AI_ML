from reversi import reversi
from collections import deque
#import time

class MiniMax:
    def minimax_Algorithm(self, game, turn, depth):#, pth=3): 
        best_score = float('-inf')
        best_move = None
        moves_queue = []

        for x in range(0, len(game.board)):
            for y in range(0, len(game.board[x])):
                if game.step(x, y, turn, False) > 0:
                    moves_queue.append((x, y))

        # Sort moves by heuristic value (if available) or random order
        moves_queue = sorted(moves_queue, key=lambda move: self.evaluate_move(game, move[0], move[1], turn), reverse=True)

        for _ in range(min(depth, len(moves_queue))):
            x, y = moves_queue.pop(0)  # Use pop(0) for list object
            score = self.min_value(game,x, y, depth - 1, -turn)
            if score > best_score:
                best_score = score
                best_move = (x, y)
        if best_move is None:
            best_move = (-1, -1)
        return best_move

    def max_value(self, game,x, y, depth, turn):
        if depth <= 0: 
            return self.evaluate_move(game, x, y, turn) #self.evaluate(game, turn)

        max_score = float('-inf')
        for x in range(0, len(game.board)):
            for y in range(0, len(game.board[x])):
                if game.step(x, y, turn, False) > 0:
                    score = self.min_value(game,x,y, depth - 1, -turn)
                    max_score = max(max_score, score)
        return max_score

    def min_value(self, game,x, y, depth, turn):
        if depth <= 0: 
            return self.evaluate_move(game, x, y, turn) #self.evaluate(game, turn) #evaluate_move(game, x, y, turn)
    
        min_score = float('inf')
        for x in range(0, len(game.board)):
            for y in range(0, len(game.board[x])):
                if game.step(x, y, turn, False) > 0:
                    score = self.max_value(game, x, y, depth - 1, -turn)
                    min_score = min(min_score, score)
        return min_score

    def evaluate(self, game, turn):
        if turn == 1:
            return game.white_count - game.black_count
        return game.black_count - game.white_count
    

    ### Think about side borders? perhaps having a higher weight?
    def evaluate_move(self, game, x, y, turn):
    # Check how many opponent pieces will be flipped by making this move
        flipped = game.step(x, y, turn, False)
    # Check the number of player's pieces adjacent to the move
        adjacent_pieces = self.count_adjacent_pieces(game, x, y, turn)
    # Evaluate the move based on a combination of flipped opponent pieces and adjacent player pieces
        return flipped - adjacent_pieces

    def count_adjacent_pieces(self, game, x, y, turn):
        adjacent_pieces = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8 and game.board[nx][ny] == turn:
                    adjacent_pieces += 1
        return adjacent_pieces


