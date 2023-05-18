from flask_restful import Resource
from ..mytypes import Game


class GamesList(Resource):
    def get(self):
        games = [
            {"id": game.value, "name": game.name.capitalize().replace("_", " ")}
            for game in Game
        ]
        return games
