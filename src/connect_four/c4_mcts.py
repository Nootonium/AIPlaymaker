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
        return len(self.children) == len(self.board.get_next_possible_moves())

    def best_child(self, c_param: Union[int, float] = 1.4) -> Optional["C4Node"]:
        best_score = float("-inf")
        best_child = None
        for child in self.children:
            if child.visits == 0:
                child_score = float("inf")
            else:
                child_score = float(
                    (child.wins / child.visits)
                    + c_param * ((2 * math.log(self.visits) / child.visits) ** 0.5)
                )
            if child_score != float("inf"):
                position = self.board.find_move_position(child.board.state)
                if child.board.blocks_opponent_win(
                    position, self.board.get_next_player()
                ):
                    child_score += 100

            if child_score > best_score:
                best_score = child_score
                best_child = child
        if best_child is None:  # TODO: replace with logging
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


class C4MCTreeSearch:
    def __init__(self, input_board: C4Board, c_param=1.4):
        self.root = C4Node(input_board)
        self.c_param = c_param

    def selection(self) -> Optional[C4Node]:
        current_node = self.root
        while current_node.fully_expanded():
            if len(current_node.board.get_next_possible_moves()) == 0:
                return None
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

    def simulation(self, node: C4Node):
        current_board = node.board
        while current_board.get_winner() is None:
            move = random.choice(current_board.get_next_possible_moves())
            current_board = current_board.with_move(move)
        winner = current_board.get_winner()
        agent_to_make_move = self.root.board.get_next_player()
        if winner == agent_to_make_move:
            return 1
        if winner == " ":
            return 0
        return -1

    def backpropagation(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent

    def run_simulation(self):
        selected_node = self.selection()
        if selected_node is None:
            return
        self.expansion(selected_node)
        result = self.simulation(selected_node)
        self.backpropagation(selected_node, result)

    def run(self, iterations):
        for _ in range(iterations):
            self.run_simulation()

        return self.root.best_child(self.c_param).board


if __name__ == "__main__":
    board = C4Board((6, 7), "11  22" + " " * 36)

    start_time = time.time()
    mcts = C4MCTreeSearch(board, 0.9)
    new_board = mcts.run(3500)
    end_time = time.time()
    print(f"Time to run MCTS at 1 thread: {end_time - start_time}")
