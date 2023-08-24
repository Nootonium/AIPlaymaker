from flask import Response, make_response, request
from flask_restful import Resource
from ..tictactoe.t3_service import T3Service


class TicTacToe(Resource):
    def __init__(self) -> None:
        super().__init__()
        self.service = T3Service()

    def post(self, action: str) -> Response:
        input_board = request.get_json().get("board")
        strategy = request.args.get("strategy")
        if input_board is None:
            return make_response({"error": "Board is required"}, 400)

        if action == "move":
            return make_response(self.service.get_next_move(input_board, strategy), 200)
        elif action == "moves":
            return make_response(
                self.service.get_next_moves(input_board, strategy), 200
            )
        else:
            return make_response({"error": "Invalid action"}, 400)
