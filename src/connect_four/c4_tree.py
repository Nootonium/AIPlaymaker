import time
from typing import Dict
from .c4_board import C4Board


class Node:
    def __init__(self, new_board: C4Board, childs=None, parents=None):
        self.board = new_board
        self.childs = childs if childs is not None else []
        self.parents = parents if parents is not None else []


class C4Dag:
    def __init__(self, input_board: C4Board) -> None:
        self.root = Node(input_board)
        self.table: Dict[str, Node] = {}
        self.build_tree()

    def build_tree(self) -> None:
        def dfs(node: Node) -> None:
            for move in node.board.get_next_possible_moves():
                new_board = node.board.with_move(move)
                if new_board.state in self.table:
                    node.childs.append(self.table[new_board.state])
                    self.table[new_board.state].parents.append(node)
                else:
                    child = Node(new_board)
                    self.table[new_board.state] = child
                    node.childs.append(child)
                    child.parents.append(node)
                    if not child.board.get_winner():
                        dfs(child)

        if not self.root.board.get_winner():
            dfs(self.root)


if __name__ == "__main__":
    board = C4Board((6, 7), "1212121" + " " * (6 * 7 - 7))
    start_time = time.time()
    dag = C4Dag(board)
    end_time = time.time()
    print(f"Time to build DAG: {end_time - start_time}")
    print(f"Number of nodes: {len(dag.table)}")
