from collections import Counter
import random
from typing import Dict

ROWS = COLS = 3


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

    def __init__(self, board="         ") -> None:
        self.root = self.Node(board)
        if self.is_valid_game_state():
            self.table: Dict[str, list] = {}
            self.build_tree()
        else:
            raise ValueError("Invalid board")

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

    def get_move_from_board(self, old_board, new_board):
        for i in range(9):
            if old_board[i] != new_board[i]:
                return i
        return None

    def build_tree(self):
        def dfs(node):
            current_player = self.get_next_player(node.val)

            for move_index in self.get_next_possible_moves(node.val):
                new_board = (
                    node.val[:move_index] + current_player + node.val[move_index + 1 :]
                )

                if new_board in self.table:
                    node.childs.append(self.table[new_board])
                else:
                    child = T3Tree.Node(new_board)
                    self.table[new_board] = child
                    node.childs.append(child)
                    if not self.game_over(new_board):
                        dfs(child)
            return

        if not self.game_over(self.root.val):
            dfs(self.root)
        return

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
        # Returns the optimal score for the current player given a game state
        state = self.game_over(node.val)
        if state == " ":
            return 0
        elif state is not None:
            return 1 if state == maxPlayer else -1

        scores = [
            self.minimax(child, not maxPlayerTurn, maxPlayer) for child in node.childs
        ]
        return max(scores) if maxPlayerTurn else min(scores)

    def get_best_next_moves(self):
        # Returns a list of the best next board states
        curr_player = self.get_next_player(self.root.val)
        """if self.root.val == "         ":
            return [
                {"move": i, "board": self.get_move_from_board(self.root.val, curr_player)} for i in range(9)
            ]"""

        scores = [
            (self.minimax(child, False, curr_player), child.val)
            for child in self.root.childs
        ]
        best_score = max(scores, key=lambda x: x[0])[0]
        best_moves = []
        for score, board in scores:
            if score == best_score:
                move = self.get_move_from_board(self.root.val, board)
                best_moves.append({"move": move, "board": board})

        return best_moves

    def get_best_next_move(self):
        next_moves = self.get_best_next_moves()
        if len(next_moves) > 1:
            return random.choice(next_moves)
        else:
            return next_moves

    def is_valid_game_state(self):
        # Check if the number of X's and O's is possible
        print(self.root.val)
        count = Counter(self.root.val)
        num_X = count.get("X", 0)
        num_O = count.get("O", 0)
        if abs(num_X - num_O) > 1:
            return False

        # Check if there is more than one winning state
        num_winning_states = sum(
            self.game_over(self.root.val) is not None for _ in self.WINNING
        )
        if num_winning_states > 1:
            return False

        return True

    def is_game_over(self):
        # Check if the game is over
        if self.game_over(self.root.val) is not None:
            return True

        # Check if there are any possible moves left
        if " " not in self.root.val:
            return True

        return False


if __name__ == "__main__":
    tree = T3Tree()
    # print(tree.get_next_best_boards())
    # print(tree.get_next_best_move())
    # print(tree.get_stats_from_childs())
    # print(tree.is_game_over())
    # print(tree.is_valid_game_state())
