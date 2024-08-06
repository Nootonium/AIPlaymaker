from flask import Response, make_response, request
from flask_restful import Resource
from ..connect_four.c4_service import C4Service


class ConnectFour(Resource):
    def __init__(self) -> None:
        super().__init__()
        self.service = C4Service()

    def post(self, action: str) -> Response:
        input_board = request.get_json().get("board")
        dimension = request.get_json().get("dimension", (6, 7))
        strategy = request.args.get("strategy")

        if input_board is None:
            return make_response({"error": "Board is required"}, 400)

        if action == "move":
            return make_response(
                self.service.get_next_move(input_board, dimension, strategy), 200
            )
        else:
            return make_response({"error": "Invalid action"}, 400)
