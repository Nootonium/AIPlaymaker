# AIPlayMaker

AIPlayMaker is a Flask-based web service designed to provide optimal next moves for various board games. Currently, the service supports games like Tic Tac Toe and Connect4, with support for Chess and more games coming soon.

AIPlayMaker is powered by intelligent algorithms and is planned to utilize Machine Learning models for more advanced game analysis in the future. It provides a simple RESTful API that allows clients to send a game board in multiple formats and receive the best possible next move or moves.

## Features

- Optimal move calculation for various games
- Supports multiple board formats
- RESTful API
- Powered by Minimax algorithm (for Tic Tac Toe)
- Utilizes Monte Carlo Tree Search (for Connect4)

## Upcoming Features

- Addition of Chess to the list of supported games
- Introduction of machine learning models to provide more advanced or faster game analysis

---

## Getting Started

To get started with AIPlayMaker, you'll need to have [Poetry](https://python-poetry.org/docs/) installed. If you don't have Poetry, you can install it with the following command:

```sh
curl -sSL https://install.python-poetry.org | python -
```

Next, clone the repository:

```sh
git clone https://github.com/yourgithubusername/aiplaymaker.git
cd aiplaymaker
```

Then, set up the virtual environment and install the required dependencies with Poetry:

```sh
poetry install
```

Before you can run the application, ensure that the `FLASK_APP` environment variable is set to your application:

```sh
export FLASK_APP=app.py
```

Finally, start the server by running the Flask application:

```sh
poetry run flask run
```

Now, AIPlayMaker will be running on `localhost:5000` (or whatever port Flask is configured to use).

---

## API Usage

AIPlayMaker provides a simple RESTful API for receiving game boards and returning optimal moves. Detailed API documentation can be found in the [GitHub wiki](link-to-wiki).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](link-to-contributing) for details on how to contribute.

## License

This project is licensed under the [GNU License](LICENSE).
