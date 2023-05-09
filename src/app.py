from flask import Flask
from flask_restful import Api
from resources.games_list import GamesList
from resources.tictactoe import TicTacToe

app = Flask(__name__)
api = Api(app)

api.add_resource(GamesList, "/games")
api.add_resource(TicTacToe, "/tictactoe/move", "/tictactoe/moves")


if __name__ == "__main__":
    app.run(debug=True)
