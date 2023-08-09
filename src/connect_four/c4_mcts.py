from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import time
import math
import random
from typing import Dict, List, Optional, Union
import numpy as np


from .c4_board import C4Board


class C4Node:
    def __init__(self, input_board: C4Board, parent: Optional["C4Node"] = None):
        self.board: C4Board = input_board
        self.parent: Optional["C4Node"] = parent
        self.children: List["C4Node"] = []
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

        position = self.board.find_move_position(child.board.state)
        if child.board.blocks_opponent_win(position, self.board.get_next_player()):
            ucb_score += 100

        return ucb_score

    def best_child(self, c_param: Union[int, float] = 1.4) -> Optional["C4Node"]:
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


def simulate_game(node: C4Node, agent_to_make_move: str):
    current_board = node.board
    while current_board.get_winner() is None:
        move = random.choice(current_board.get_next_possible_moves())
        current_board = current_board.with_move(move)
    winner = current_board.get_winner()
    if winner == agent_to_make_move:
        return 1
    if winner == " ":
        return 0
    return -1


class C4MCTreeSearch:
    def __init__(self, input_board: C4Board, c_param=1.4):
        self.root = C4Node(input_board)
        self.c_param = c_param
        self.backpropagation_lock = Lock()

    def selection(self) -> Optional[C4Node]:
        current_node = self.root
        while current_node.fully_expanded():
            node = current_node.best_child(self.c_param)
            if node is None:  # to satisfy mypy
                return None
            current_node = node
        return current_node

    def expansion(self, node: C4Node):
        possible_moves = node.board.get_next_possible_moves()
        for move in possible_moves:
            next_board = node.board.with_move(move)
            child_node = C4Node(next_board, node)
            node.add_child(child_node)

    def backpropagation(self, node, result):
        with self.backpropagation_lock:
            while node is not None:
                node.visits += 1
                node.wins += result
                node = node.parent

    def run(self, iterations, num_threads=4):
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(iterations):
                selected_node = self.selection()
                if selected_node is None:
                    continue
                self.expansion(selected_node)

                # Use executor to run simulation in parallel
                future = executor.submit(
                    simulate_game, selected_node, self.root.board.get_next_player()
                )

                # Wait for the simulation result and perform backpropagation
                result = future.result()
                self.backpropagation(selected_node, result)

        return self.root.best_child(self.c_param).board


if __name__ == "__main__":
    board = C4Board((6, 7), "11  22" + " " * 36)

    start_time = time.time()
    mcts = C4MCTreeSearch(board, 0.9)

    new_board = mcts.run(250)

    end_time = time.time()
    print(f"Time to run MCTS at 1 thread: {end_time - start_time}")
