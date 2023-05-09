from collections import Counter
import random

ROWS = COLS = 3

# Definition for a Node.


def is_board_valid(board):
    moveset = set(["X", "O", " "])

    if len(board) != 9:
        return False
    for move in board:
        if move not in moveset:
            return False
    return True


class T3Tree:
    class Node:
        def __init__(self, val, childs=None):
            self.val = val
            self.childs = childs if childs is not None else []

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

    def __init__(self, board=[[" " for _ in range(ROWS)] for _ in range(COLS)]) -> None:
        self.root = self.Node(board)
        self.table = {}
        self.populate()

    def game_over(self, board):
        gameState = list(board)
        for wins in self.WINNING:
            # Create a tuple
            w = (gameState[wins[0]], gameState[wins[1]], gameState[wins[2]])
            if w == ("X", "X", "X"):
                return "X"
            if w == ("O", "O", "O"):
                return "O"
        # Check for stalemate
        if " " in gameState:
            return None
        return " "

    def get_next_player(self, board):
        count = Counter(board)
        return "X" if count.get("X", 0) <= count.get("O", 0) else "O"

    def get_next_possible_moves(self, board):
        gameState = list(board)
        return [i for i, p in enumerate(gameState) if p == " "]

    def build_tree(self):
        def dfs(node):
            current_player = self.get_next_player(node.val)

            for move_index in self.get_next_possible_moves(node.val):
                new_board = (
                    node.val[:move_index] + current_player + node.val[move_index + 1 :]
                )

                if new_board in self.table:
                    node.children.append(self.table[new_board])
                else:
                    child = self.Node(new_board)
                    self.table[new_board] = child
                    node.children.append(child)
                    if not self.game_over(new_board):
                        dfs(child)
            return

        if not self.game_over(self.root.val):
            dfs(self.root)

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

    def minimax(self, node, maxPlayerTurn, maxPlayer):
        state = self.game_over(node.val)
        if state == " ":
            return 0
        elif state is not None:
            return 1 if state == maxPlayer else -1

        scores = []
        for child in node.childs:
            scores.append(self.minimax(child, not (maxPlayerTurn), maxPlayer))
        return max(scores) if maxPlayerTurn else min(scores)

    def get_best_next_moves(self):
        if self.game_over(self.root.val):
            raise Exception("The game is finished")

        if self.root.val == "         ":
            return [x for x in range(9)]

        self.populate  # find a way this is only called once

        def minimax(node, maxPlayerTurn, maxPlayer):
            state = self.game_over(node.val)
            if state == " ":
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
        moves = self.get_next_possible_moves(self.root.val)
        for score, i in scores:
            if score == max_move:
                ans.append(moves[i])

        return ans

    def get_best_next_move(self):
        next_moves = self.get_best_next_moves()
        if len(next_moves) > 1:
            return random.choice(next_moves)
        else:
            return next_moves
