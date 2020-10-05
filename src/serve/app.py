from logging import getLogger

from flask import Flask, request, jsonify

from ..config import PROJECT_NAME
from .parser import parse_request_body
from ..predict.main import predict

PING_ROUTE = '/ping'
PREDICT_ROUTE = '/predict'
DEBUG_PORT = 5000

logger = getLogger(__name__)

app = Flask(PROJECT_NAME)


@app.route(PING_ROUTE, methods=['GET'])
def ping_view():
    return 'pong!'


@app.route(PREDICT_ROUTE, methods=['POST'])
def predict_view():
    feed, error_msg = parse_request_body(request)
    if error_msg:
        logger.warning(f'Client error: {error_msg}')
        return jsonify({'message': error_msg}), 400

    try:
        predictions = predict(feed)
    except Exception as e:
        msg = f'An error occured during prediction: {e}'
        logger.error(msg)
        return jsonify({'message': msg}), 500

    return jsonify(predictions.tolist())


def main():
    # The `run()` method is just a convenient webserver to use for debugging.
    # It is absolutely *NOT* suited for production pruposes.
    app.run(port=DEBUG_PORT, debug=True, host='0.0.0.0')


if __name__ == '__main__':
    main()
