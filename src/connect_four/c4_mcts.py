import time
import random
from typing import List
from .c4_board import C4Board


class C4Node:
    def __init__(self, board: C4Board, parent=None):
        self.board = board
        self.parent = parent
        self.children: List[C4Node] = []
        self.wins = 0
        self.visits = 0


class C4MCTS:
    def __init__(self, root: C4Node):
        self.root = root

    def select(self):
        """Selects a leaf node to expand."""
        node = self.root
        while node.children:
            node = random.choice(node.children)  # TODO: Use a smarter policy.
        return node

    def expand(self, node: C4Node):
        """Expands a node by generating all possible next moves."""
        next_player = node.board.get_next_player()
        next_moves = node.board.get_next_possible_moves()
        print(next_moves)
        for move in next_moves:
            new_state = (
                node.board.state[:move] + next_player + node.board.state[move + 1 :]
            )
            new_board = C4Board(node.board.dimensions, new_state)
            node.children.append(C4Node(new_board, parent=node))

    def simulate(self, node: C4Node):
        """Simulates a game until the end and returns the winner."""
        current_board = node.board
        while current_board.get_winner() is None:
            player = current_board.get_next_player()
            moves = current_board.get_next_possible_moves()
            move = random.choice(moves)  # TODO: Use a smarter policy.
            new_state = (
                current_board.state[:move] + player + current_board.state[move + 1 :]
            )
            current_board = C4Board(current_board.dimensions, new_state)
        return current_board.get_winner()

    def backpropagate(self, node: C4Node, winner: str):
        """Backpropagates the result of a simulation up the tree."""
        while node is not None:
            node.visits += 1
            if node.board.get_next_player() == winner:
                node.wins += 1
            node = node.parent

    def run_simulation(self):
        node = self.select()
        if node.board.get_winner() is None:
            self.expand(node)
            node = random.choice(node.children)  # TODO: Use a smarter policy.
        winner = self.simulate(node)
        self.backpropagate(node, winner)

    def get_best_move(self):
        best_child = max(self.root.children, key=lambda c: c.wins / c.visits)
        return self.root.board.find_move_position(best_child.board.state)


if __name__ == "__main__":
    board = C4Board((6, 7), " " * 42)
    root = C4Node(board)
    mcts = C4MCTS(root)

    start_time = time.time()  # Record the start time
    for _ in range(1):
        mcts.run_simulation()
    end_time = time.time()  # Record the end time

    print(mcts.get_best_move())
    duration = end_time - start_time  # Calculate the duration
    print(f"The simulations took {duration} seconds.")
