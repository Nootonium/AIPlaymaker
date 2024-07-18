from typing import List, Optional, Dict, Union
import numpy as np
import math

from .board import BoardState, Position, MoveType, GameBoard


class Node:
    def __init__(self, input_board: GameBoard, parent: Optional["Node"] = None):
        self.board: GameBoard = input_board
        self.parent: Optional["Node"] = parent
        self.children: List["Node"] = []
        self.wins: int = 0
        self.visits: int = 0

    def add_child(self, child_node) -> None:
        self.children.append(child_node)

    def update(self, result) -> None:
        self.visits += 1
        self.wins += result

    def fully_expanded(self) -> bool:
        moves = len(self.board.get_next_possible_moves())
        return len(self.children) == moves and moves > 0

    def _child_score(self, child, c_param):
        if child.visits == 0:
            return float("inf")

        ucb_score = (child.wins / child.visits) + c_param * (
            (2 * math.log(self.visits) / child.visits) ** 0.5
        )
        return ucb_score

    def best_child(self, c_param: Union[int, float] = 1.4) -> Optional["Node"]:
        best_child = max(
            self.children,
            key=lambda child: self._child_score(child, c_param),
            default=None,
        )

        if best_child is None:
            message = len(self.children)
            raise Exception("No best child found. Children: " + str(message))

        return best_child

    def get_q_values(self) -> Dict[int, float]:
        q_values = {}
        for child in self.children:
            _, action = self.board.find_move_position(child.board.state)
            q_value = 0.0
            if child.visits > 0:
                q_value = float(child.wins) / child.visits
            q_values[action] = q_value
        return q_values

    def get_probs(self, temperature=1) -> Dict[int, float]:
        q_values = self.get_q_values()
        values = np.array(list(q_values.values()))
        values /= temperature

        # Softmax function for converting Q-values to probabilities
        probs = np.exp(values) / np.sum(np.exp(values))

        return dict(zip(q_values.keys(), probs))


def simulate_random_game(node: Node) -> GameBoard:
    current_board = node.board
    while not current_board.is_game_complete():
        possible_moves = current_board.get_next_possible_moves()
        random_move = possible_moves[np.random.randint(len(possible_moves))]
        current_board = current_board.make_move(
            random_move["start"], random_move["target"]
        )
    return current_board


class MCTS:
    def __init__(self, root: Node, c_param: Union[int, float] = 1.4):
        self.root: Node = root
        self.c_param: Union[int, float] = c_param

    def search(self, num_simulations: int) -> Node:
        for _ in range(num_simulations):
            node = self.root
            while node.fully_expanded() and node.children:
                node = node.best_child(self.c_param)

            if not node.fully_expanded():
                possible_moves = node.board.get_next_possible_moves()
                random_move = possible_moves[np.random.randint(len(possible_moves))]
                new_board = node.board.make_move(
                    random_move["start"], random_move["target"]
                )
                new_node = Node(new_board, parent=node)
                node.add_child(new_node)
                node = new_node

            winner = simulate_random_game(node).get_winner()
            result = 1 if winner == node.board.current_turn else 0
            while node is not None:
                node.update(result)
                node = node.parent

        return self.root.best_child(c_param=0)
