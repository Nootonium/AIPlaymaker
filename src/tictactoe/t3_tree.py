import random
from typing import Any, Dict

from .t3_board import T3Board


class T3Tree:
    class Node:
        def __init__(self, board, childs=None):
            self.board = board
            self.childs = childs if childs is not None else []

    def __init__(self, board: T3Board) -> None:
        self.root = self.Node(board)
        if self.root.board.is_valid_game_state():
            self.table = {}
            self.build_tree()
        else:
            raise ValueError("Invalid board")

    def get_move_from_board(self, old_board, new_board) -> int | None:
        for i in range(9):
            if old_board[i] != new_board[i]:
                return i
        return None

    def build_tree(self) -> None:
        def dfs(node):
            current_player = node.board.get_next_player()

            for move_index in node.board.get_next_possible_moves():
                new_board = (
                    node.val[:move_index] + current_player + node.val[move_index + 1 :]
                )

                if new_board in self.table:
                    node.childs.append(self.table[new_board])
                else:
                    child = T3Tree.Node(new_board)
                    self.table[new_board] = child
                    node.childs.append(child)
                    if not child.board.game_over():
                        dfs(child)

        if not self.root.board.game_over():
            dfs(self.root)

    def get_stats_from_childs(self):
        res = []

        def dfs(node, curr_stat):
            game_over = node.board.game_over()

            if game_over:
                curr_stat[game_over] = curr_stat.get(game_over, 0) + 1

            for child in node.childs:
                dfs(child, curr_stat)
            return curr_stat

        for child in self.root.childs:
            stat = dfs(child, {})
            res.append(stat)
        return res

    def minimax(self, node, max_player_turn, max_player) -> int:
        # Returns the optimal score for the current player given a game state
        state = node.board.game_over()
        if state == " ":
            return 0
        if state is not None:
            return 1 if state == max_player else -1

        scores = [
            self.minimax(child, not max_player_turn, max_player)
            for child in node.childs
        ]
        return max(scores) if max_player_turn else min(scores)

    def get_best_next_moves(self) -> list:
        # Returns a list of the best next board states
        curr_player = self.root.board.get_next_player()
        # TODO: return all moves if first move

        scores = [
            (self.minimax(child, False, curr_player), child.val)
            for child in self.root.childs
        ]
        best_score = max(scores, key=lambda x: x[0])[0]
        best_moves = []
        for score, new_board in scores:
            if score == best_score:
                move = self.get_move_from_board(self.root.board, new_board)
                best_moves.append({"move": move, "board": new_board})

        return best_moves

    def get_best_next_move(self) -> Dict[Any, Any]:
        next_moves = self.get_best_next_moves()
        if len(next_moves) > 1:
            return random.choice(next_moves)
        else:
            return next_moves


if __name__ == "__main__":
    board = "         "
    tree = T3Tree(T3Board(board))
    print(tree.get_best_next_moves())
