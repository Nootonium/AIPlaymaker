from unittest import TestCase
from src.tictactoe.t3_board import T3Board
from src.tictactoe.t3_tree import T3Tree, Node
from .test_constants import (
    VALID_BOARD,
    COMMON_BOARD_OPENING,
    X_ALMOST_WINS,
    O_ALMOST_WINS,
    VALID_O_WINS,
    EMPTY_BOARD,
)


class TestT3Tree(TestCase):
    def setUp(self):
        self.board = T3Board(VALID_BOARD)
        self.tree = T3Tree(self.board)

    def test_init(self):
        self.assertIsInstance(self.tree.root, Node)
        self.assertEqual(self.tree.root.board, self.board)

    def test_tree_properties_after_construction(self):
        # Given
        initial_state = T3Board(COMMON_BOARD_OPENING)
        tree = T3Tree(initial_state)

        # When
        root_state = tree.root.board.state
        root_child_count = len(tree.root.childs)

        # Then
        self.assertEqual(
            root_state,
            initial_state.state,
            "Root state of the tree should match the initial state",
        )
        self.assertEqual(
            root_child_count, 8, "Root should have 8 children (for 8 possible moves)"
        )
        for child_node in tree.root.childs:
            self.assertEqual(
                len(child_node.board.state),
                9,
                "Each child node should have a board state of length 9",
            )
            if not child_node.board.get_winner():
                self.assertGreater(
                    len(child_node.childs),
                    0,
                    "Non-terminal child nodes should have children",
                )
            else:
                self.assertEqual(
                    len(child_node.childs),
                    0,
                    "Terminal child nodes should not have children",
                )

    def test_given_x_almost_win_when_minimax_then_return_1(self):
        initial_state = T3Board(X_ALMOST_WINS)
        tree = T3Tree(initial_state)

        score = tree.minimax(tree.root, True, "X")

        self.assertEqual(score, 1)

    def test_given_board_when_minimax_then_return_0(self):
        initial_state = T3Board(VALID_BOARD)
        tree = T3Tree(initial_state)

        score = tree.minimax(tree.root, True, "O")

        self.assertEqual(score, 0)

    def test_given_valid_o_win_when_minimax_then_return_minus_1(self):
        initial_state = T3Board(VALID_O_WINS)
        tree = T3Tree(initial_state)

        score = tree.minimax(tree.root, True, "X")

        self.assertEqual(score, -1)

    def test_given_empty_board_when_minimax_then_return_valid_score(self):
        initial_state = T3Board(EMPTY_BOARD)
        tree = T3Tree(initial_state)

        score = tree.minimax(tree.root, True, "X")

        self.assertTrue(-1 <= score <= 1)

    def test_given_known_board_when_get_scores_then_return_expected_scores(self):
        initial_state = T3Board(O_ALMOST_WINS)
        tree = T3Tree(initial_state)

        scores = tree.get_scores()

        expected_scores = [1, 0, -1, -1]
        actual_scores = [score for score, _ in scores]
        self.assertEqual(actual_scores, expected_scores)

    def test_given_known_scores_when_get_best_moves_then_return_expected_moves(self):
        pass

    def test_get_stats_from_childs(self):
        # Here you can test get_stats_from_childs method
        pass

    def test_count_leafs(self):
        # Here you can test count_leafs method
        pass


if __name__ == "__main__":
    pass
