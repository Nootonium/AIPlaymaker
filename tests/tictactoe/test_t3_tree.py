import unittest
from src.tictactoe.t3_tree import T3Tree
from src.tictactoe.t3_board import T3Board


class TestT3Tree(unittest.TestCase):
    def setUp(self):
        self.board = T3Board("XO       ")
        self.tree = T3Tree(self.board)

    def test_get_best_next_moves(self):
        moves = self.tree.get_best_next_moves()
        self.assertIsInstance(moves, list)
        for move in moves:
            self.assertIn("move", move)
            self.assertIn("post_move_board", move)

    def test_get_move_from_board(self):
        old_board = "         "
        new_board = "X        "
        move = self.tree.get_move_from_board(old_board, new_board)
        self.assertEqual(move, 0)

    def test_minimax(self):
        score = self.tree.minimax(self.tree.root, True, "X")
        self.assertIsInstance(score, int)

    def test_get_best_next_move(self):
        move = self.tree.get_best_next_move()
        self.assertIn("move", move)
        self.assertIn("post_move_board", move)


if __name__ == "__main__":
    unittest.main()
