import time
import math
import random
from .c4_board import C4Board


class C4Node:
    def __init__(self, input_board: C4Board, parent=None):
        self.board = input_board
        self.parent = parent
        self.children: list = []
        self.wins = 0
        self.visits = 0

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, result):
        self.visits += 1
        self.wins += result

    def fully_expanded(self):
        return len(self.children) == len(self.board.get_next_possible_moves())

    def best_child(self, c_param=1.4):
        best_score = -1
        best_child = None
        for child in self.children:
            if child.visits == 0:
                child_score = float("inf")
            else:
                child_score = (child.wins / child.visits) + c_param * (
                    (2 * math.log(self.visits) / child.visits) ** 0.5
                )
            if child_score > best_score:
                best_score = child_score
                best_child = child
        return best_child


class C4MCTS:
    def __init__(self, node: C4Node):
        self.root = node

    def selection(self):
        current_node = self.root
        while current_node.fully_expanded():
            current_node = current_node.best_child()
        return current_node

    def expansion(self, node):
        possible_moves = node.board.get_next_possible_moves()
        for move in possible_moves:
            next_board = node.board.with_move(move, node.board.get_next_player())
            child_node = C4Node(next_board, node)
            node.add_child(child_node)

    def simulation(self, node: C4Node):
        current_board = node.board
        while not (current_board.get_winner() is not None):
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

    def run(self, iterations):
        for _ in range(iterations):
            selected_node = self.selection()
            self.expansion(selected_node)
            result = self.simulation(selected_node)
            self.backpropagation(selected_node, result)
        return self.root.best_child().board


if __name__ == "__main__":
    board = C4Board((6, 7), "11  22" + " " * 36)
    root = C4Node(board)
    mcts = C4MCTS(root)
    start_time = time.time()
    new_board = mcts.run(1000)
    end_time = time.time()
    print("Time to run: ", end_time - start_time)
    print(board.find_move_position(new_board.state))
    print(new_board.state.replace(" ", "_"))
    for child in root.children:
        print(
            "Board:",
            child.board.state.replace(" ", "_"),
            "wins:",
            child.wins,
            " visits:",
            child.visits,
        )
