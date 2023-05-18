"""TicTacToe resource module."""

from flask import request
from flask_restful import Resource
from ..tictactoe.t3_service import T3Service


class TicTacToe(Resource):
    def __init__(self) -> None:
        super().__init__()
        self.service = T3Service()

    def post(self) -> list | dict:
        input_board = request.get_json()["board"]

        if request.path == "/tictactoe/move":
            return self.service.get_next_move(input_board)
        else:
            return self.service.get_next_moves(input_board)
