from flask import Flask, jsonify
from flask_restful import Api
from resources.games_list import GamesList

from src.resources.tictactoe import TicTacToe
from src.exceptions import InvalidBoardFormatError

app = Flask(__name__)
api = Api(app)

api.add_resource(GamesList, "/games")
api.add_resource(TicTacToe, "/tictactoe/move", "/tictactoe/moves")


@app.errorhandler(InvalidBoardFormatError)
def handle_invalid_usage(error):
    response = jsonify({"message": str(error)})
    response.status_code = 400
    return response


if __name__ == "__main__":
    app.run(debug=True)
