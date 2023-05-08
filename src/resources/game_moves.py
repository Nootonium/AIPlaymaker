from flask_restful import Resource
from flask import request
from mytypes import Game

class GameMoves(Resource):
    def post(self, game_id):
        if game_id == Game.TICTACTOE.value:
            return {"next_move": "B2"}

        elif game_id == Game.CHESS.value:
            return {"next_move": "E4"}

        else:
            return {"error": "Unsupported game"}, 400