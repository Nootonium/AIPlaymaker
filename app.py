from flask import Flask, jsonify
from flask_restful import Api
from dotenv import load_dotenv

from src.resources.games_list import GamesList
from src.resources.tictactoe import TicTacToe
from src.exceptions import InvalidBoardException
from config import os, Development, Staging, Production

load_dotenv()

app = Flask(__name__)

match os.getenv("FLASK_ENV", "development"):
    case "development":
        app.config.from_object(Development)
    case "staging":
        app.config.from_object(Staging)
    case "production":
        app.config.from_object(Production)
    case _:
        raise ValueError("Invalid FLASK_ENV: how did you fuck this up?")

api = Api(app)

api.add_resource(GamesList, "/games")
api.add_resource(TicTacToe, "/tictactoe/move", "/tictactoe/moves")


@app.errorhandler(InvalidBoardException)
def handle_invalid_usage(error):
    response = jsonify({"message": str(error)})
    response.status_code = 400
    return response


if __name__ == "__main__":
    app.run()
