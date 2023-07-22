import time
import math
import random
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor


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
            if child_score > best_score:
                best_score = child_score
                best_child = child
        if best_child is None:  # TODO: replace with logging
            message = len(self.children)
            raise Exception("No best child found. Children: " + str(message))
        return best_child


class C4MCTreeSearch:
    def __init__(self, input_board: C4Board):
        self.root = C4Node(input_board)

    def selection(self) -> Optional[C4Node]:
        current_node = self.root
        while current_node.fully_expanded():
            if len(current_node.board.get_next_possible_moves()) == 0:
                return None
            node = current_node.best_child()
            if node is None:  # to satisfy mypy
                return None
            current_node = node
        return current_node

    def expansion(self, node: C4Node):
        possible_moves = node.board.get_next_possible_moves()
        for move in possible_moves:
            next_board = node.board.with_move(move, node.board.get_next_player())
            child_node = C4Node(next_board, node)
            node.add_child(child_node)

    def simulation(self, node: C4Node):
        current_board = node.board
        while current_board.get_winner() is None:
            move = random.choice(current_board.get_next_possible_moves())
            current_board = current_board.with_move(
                move, current_board.get_next_player()
            )
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

    def run(self, iterations, num_threads=4):
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(iterations):
                executor.submit(self.run_simulation)
        return self.root.best_child().board


if __name__ == "__main__":
    board = C4Board((6, 7), "11  22" + " " * 36)
    mcts = C4MCTreeSearch(board)
    start_time = time.time()
    new_board = mcts.run(1000)
    end_time = time.time()
    print("Time to run: ", end_time - start_time)
    print(board.find_move_position(new_board.state))
    print(new_board.state.replace(" ", "_"))
    """for child in mcts.root.children:
        print(
            "Board:",
            child.board.state.replace(" ", "_"),
            "wins:",
            child.wins,
            " visits:",
            child.visits,
        )"""
