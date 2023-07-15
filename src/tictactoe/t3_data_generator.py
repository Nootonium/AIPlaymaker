from .t3_board import T3Board
from .t3_tree import T3Tree, Node


class DataGenerator:
    def __init__(self):
        self.training_data = []
        self.full_tree = T3Tree(T3Board(" " * 9))

    def generate_data(self):
        for child in self.full_tree.root.childs:
            self._dfs(child)

    def _dfs(self, node: Node):
        game_state = node.board.get_winner()
        if game_state is not None or game_state == " ":
            return

        for child in node.childs:
            self._dfs(child)

        new_root = self.full_tree.get_tree_from_board(node.board)

        current_tree = T3Tree.from_root(new_root)
        best_next_moves = current_tree.get_best_next_moves()
        next_moves = []
        for move, _ in best_next_moves:
            next_moves.append(move)
        self.training_data.append((node.board.state, next_moves))

    def get_training_data(self):
        return self.training_data


if __name__ == "__main__":
    generator = DataGenerator()
    generator.generate_data()
    print(generator.get_training_data()[0])
    print(len(generator.get_training_data()))
