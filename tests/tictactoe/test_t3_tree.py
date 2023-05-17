import unittest
from src.tictactoe.t3_board import T3Board
from src.tictactoe.t3_tree import T3Tree, Node


class TestT3Tree(unittest.TestCase):
    midle_X_start = "O   X    "  # Todo: refactor test constants

    def setUp(self):
        self.board = T3Board(self.midle_X_start)
        self.tree = T3Tree(self.board)

    def test_init(self):
        self.assertIsInstance(self.tree.root, Node)
        self.assertEqual(self.tree.root.board, self.board)

    def test_get_move_from_board(self):
        # Here you can test get_move_from_board method
        pass

    def test_build_tree(self):
        # Here you can test build_tree method
        pass

    def test_minimax(self):
        # Here you can test minimax method
        pass

    def test_get_scores(self):
        # Here you can test get_scores method
        pass

    def test_get_best_score(self):
        # Here you can test get_best_score method
        pass

    def test_get_best_moves(self):
        # Here you can test get_best_moves method
        pass

    def test_get_best_next_moves(self):
        # Here you can test get_best_next_moves method
        pass

    def test_get_best_next_move(self):
        # Here you can test get_best_next_move method
        pass

    def test_get_stats_from_childs(self):
        # Here you can test get_stats_from_childs method
        pass

    def test_count_leafs(self):
        # Here you can test count_leafs method
        pass


if __name__ == "__main__":
    unittest.main()
