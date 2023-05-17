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

    def test_given_two_diff_moves_when_find_difference_position_then_raises_value_error(
        self,
    ):
        old_board = "         "
        new_board = "XO       "

        with self.assertRaises(ValueError):
            T3Tree.find_difference_position(self, old_board, new_board)

    def test_given_one_diff_move_when_find_difference_position_then_returns_index(self):
        old_board = "         "
        new_board = "X        "

        index = T3Tree.find_difference_position(self, old_board, new_board)

        self.assertEqual(index, 0)

    def test_given_no_diff_moves_when_find_difference_position_then_returns_none(self):
        old_board = "         "
        new_board = "         "

        index = T3Tree.find_difference_position(self, old_board, new_board)

        self.assertIsNone(index)

    def test_build_tree(self):
        initial_board_state = "    X    "
        board = T3Board(initial_board_state)
        tree = T3Tree(board)

        best_next_moves = tree.get_best_next_moves()

        self.assertEqual(len(best_next_moves), 4)

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
