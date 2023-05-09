from collections import deque, Counter
import copy

ROWS = COLS = 3

# Definition for a Node.


def is_board_valid(board):
    moveset = set(['X', 'O', ' '])

    if len(board) != 9:
        return False
    for move in board:
        if move not in moveset:
            return False
    return True


class Node:
    def __init__(self, val, childs=None):
        self.val = val
        self.childs = childs if childs is not None else []


class T3Tree():
    WINNING = [
        [0, 1, 2],  # Across top
        [3, 4, 5],  # Across middle
        [6, 7, 8],  # Across bottom
        [0, 3, 6],  # Down left
        [1, 4, 7],  # Down middle
        [2, 5, 8],  # Down right
        [0, 4, 8],  # Diagonal ltr
        [2, 4, 6],  # Diagonal rtl
    ]

    def __init__(self, board=[[" " for _ in range(ROWS)]for _ in range(COLS)]) -> None:
        if is_board_valid(board):
            self.root = Node(board)
        else:
            raise ValueError("Could not create tree. Board is invalid")
        self.table = {}

    def game_over(self, board):
        gameState = list(board)
        for wins in self.WINNING:
            # Create a tuple
            w = (gameState[wins[0]], gameState[wins[1]], gameState[wins[2]])
            if w == ('X', 'X', 'X'):
                return 'X'
            if w == ('O', 'O', 'O'):
                return 'O'
        # Check for stalemate
        if ' ' in gameState:
            return None
        return ' '

    def get_next_player(self, board):
        count = Counter(board)
        return 'X' if count.get('X', 0) <= count.get('O', 0) else "O"

    def get_next_moves(self, board):
        gameState = list(board)
        return [i for i, p in enumerate(gameState) if p == " "]

    @property
    def populate(self):
        def dfs(node):
            curr_player = self.get_next_player(node.val)

            for move_index in self.get_next_moves(node.val):

                new_board = node.val[:move_index] + \
                    curr_player + node.val[move_index + 1:]

                if new_board in self.table:
                    node.childs.append(self.table[new_board])
                else:
                    child = Node(new_board)
                    self.table[new_board] = child
                    node.childs.append(child)
                    if not self.game_over(new_board):
                        dfs(child)
            return

        node = self.root
        if not (self.game_over(node.val)):
            dfs(node)
        return

    @property
    def get_stats_from_childs(self):
        res = []

        def dfs(node, curr_stat):
            game_over = self.game_over(node.val)

            if game_over:
                curr_stat[game_over] = curr_stat.get(game_over, 0) + 1

            for child in node.childs:
                dfs(child, curr_stat)
            return curr_stat

        for child in self.root.childs:
            stat = dfs(child, {})
            res.append(stat)
        return res

    @property
    def get_best_next_moves(self):
        if self.game_over(self.root.val):
            raise Exception("The game is finished")
        
        if self.root.val == "         ":
            return [x for x in range(9)]
        
        self.populate # find a way this is only called once
        
        def minimax(node, maxPlayerTurn, maxPlayer):
            state = self.game_over(node.val)
            if state == ' ':
                return 0
            elif state is not None:
                return 1 if state == maxPlayer else -1

            scores = []
            for child in node.childs:
                scores.append(minimax(child, not (maxPlayerTurn), maxPlayer))
            return max(scores) if maxPlayerTurn else min(scores)

        scores = []
        curr_player = self.get_next_player(self.root.val)
        for i, child in enumerate(self.root.childs):
            score = minimax(child, False, curr_player)
            scores.append((score, i))
        max_move = max(scores)[0]
        ans = []
        moves = self.get_next_moves(self.root.val)
        for score, i in scores:
            if score == max_move:
                ans.append(moves[i])
                
        return ans