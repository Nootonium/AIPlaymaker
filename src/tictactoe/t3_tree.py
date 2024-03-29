import random
from typing import Dict, List, Tuple, Type

from .t3_board import T3Board


class Node:
    def __init__(self, new_board: T3Board, childs=None):
        self.board = new_board
        self.childs = childs if childs is not None else []


class T3Tree:
    def __init__(self, input_board: T3Board, build_tree: bool = True) -> None:
        self.root = Node(input_board)
        self.table: Dict[str, Type["Node"]] = {}
        if build_tree:
            if self.root.board.is_valid_game_state():
                self.build_tree()
            else:
                raise ValueError("Invalid board")

    @classmethod
    def from_root(cls, root_node: Node) -> "T3Tree":
        new_tree = cls(root_node.board, build_tree=False)
        new_tree.root.childs = root_node.childs
        return new_tree

    def build_tree(self) -> None:
        def dfs(node):
            current_player = node.board.get_next_player()

            for move_index in node.board.get_next_possible_moves():
                new_board = (
                    node.board.state[:move_index]
                    + current_player
                    + node.board.state[move_index + 1 :]
                )
                new_board = T3Board(new_board)
                if new_board.state in self.table:
                    node.childs.append(self.table[new_board.state])
                else:
                    child = Node(new_board)
                    self.table[new_board.state] = child
                    node.childs.append(child)
                    if not child.board.get_winner():
                        dfs(child)

        if not self.root.board.get_winner():
            dfs(self.root)

    def get_tree_from_board(self, board) -> Type["Node"] | None:
        return self.table.get(board.state)

    def minimax(self, node: Type["Node"], max_player_turn, max_player) -> int:
        state = node.board.get_winner()
        if state == " ":
            return 0
        if state is not None:
            return 1 if state == max_player else -1

        scores = [
            self.minimax(child, not max_player_turn, max_player)
            for child in node.childs
        ]

        return max(scores) if max_player_turn else min(scores)

    def get_scores(self) -> List[Tuple[int, Type["Node"]]]:
        curr_player = self.root.board.get_next_player()
        return [
            (self.minimax(child, False, curr_player), child)
            for child in self.root.childs
        ]

    def get_best_score(self, scores: List[Tuple[int, Type["Node"]]]) -> int:
        return max(scores, key=lambda x: x[0])[0]

    def get_best_moves(
        self, best_score: int, scores: List[Tuple[int, Type["Node"]]]
    ) -> List[Tuple[int, str]]:
        best_moves = []
        for score, node in scores:
            if score == best_score:
                move = self.root.board.find_move_position(node.board.state)
                if move is not None:
                    best_moves.append(
                        (
                            move,
                            node.board.state,
                        )
                    )
        return best_moves

    def get_best_next_moves(
        self,
    ) -> List[Tuple[int, str]]:
        scores = self.get_scores()
        best_score = self.get_best_score(scores)
        return self.get_best_moves(best_score, scores)

    def get_best_next_move(
        self,
    ) -> Tuple[int, str]:
        next_moves = self.get_best_next_moves()

        return random.choice(next_moves)

    def get_stats_from_childs(self):
        res = []

        def dfs(node, curr_stat):
            game_over = node.board.get_winner()

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
    def count_leafs(self):
        return self._count_leafs(self.root)

    @classmethod
    def _count_leafs(cls, node: Type["Node"]):
        return (
            sum(cls._count_leafs(child) for child in node.childs) if node.childs else 1
        )


if __name__ == "__main__":
    BOARD = T3Board("    X    ")
    if BOARD.get_winner() is None:
        tree = T3Tree(BOARD)
        print(tree.get_best_next_moves())
        print(tree.get_best_next_move())
        print(len(tree.table))
        print(tree.count_leafs)
    else:
        print("Game over")
