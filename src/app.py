import os
from flask import Flask
from flask_restful import Api
from resources.games_list import GamesList
from resources.game_moves import GameMoves

app = Flask(__name__)

if os.environ.get("FLASK_ENV") == "development":
    app.debug = True
else:
    app.debug = False


api = Api(app)

api.add_resource(GamesList, "/games")
api.add_resource(GameMoves, "/games/<string:game_id>/moves")


if __name__ == "__main__":
    app.run(debug=app.debug)
