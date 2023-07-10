from .t3_board import T3Board
from .t3_tree import T3Tree, Node


class DataGenerator:
    def __init__(self):
        self.training_data = []
        self.full_tree = T3Tree(T3Board(" " * 9))

    def generate_data(self):
        empty_board = T3Board(" " * 9)
        full_tree = T3Tree(empty_board)
        self._dfs(full_tree.root)

    def _dfs(self, node: Node):
        game_state = node.board.get_winner()
        if game_state is not None or game_state == " ":
            return

        for child in node.childs:
            self._dfs(child)

        current_tree = T3Tree.from_root(self.full_tree.get_tree_from_board(node.board))
        best_next_moves = current_tree.get_best_next_moves()
        next_moves = []
        for play in best_next_moves:
            next_moves.append(play.get("move"))
        if next_moves:
            print(node.board.state, next_moves)
            input("Press Enter to continue...")
            self.training_data.append((node.board.state, next_moves))
        else:
            print("No moves")

    def get_training_data(self):
        return self.training_data


if __name__ == "__main__":
    generator = DataGenerator()
    generator.generate_data()
