from flask import Flask, request
from flask_restful import Resource, Api
from resources.games_list import GamesList
from resources.game_moves import GameMoves

app = Flask(__name__)
api = Api(app)

api.add_resource(GamesList, "/games")
api.add_resource(GameMoves, "/games/<string:game_id>/moves")


if __name__ == "__main__":
    app.run(debug=True)