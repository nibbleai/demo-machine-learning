from unittest.mock import patch, Mock

import pytest

from src.serve.app import app as flask_app, PING_ROUTE, PREDICT_ROUTE

flask_app.config['TESTING'] = True

# A regular prediction is a numpy array, transformed to a list by the
# Flask application's view.
FAKE_PREDICTION = Mock()
FAKE_PREDICTION.tolist.return_value = [1, 2, 3]


@pytest.fixture
def client():
    with flask_app.test_client() as client:
        yield client


def test_ping(client):
    resp = client.get(PING_ROUTE)

    assert resp.status_code == 200


@patch('src.serve.app.predict')
@patch('src.serve.app.parse_request_body')
def test_predict_view_calls_predict(mock_parser, mock_predict, client):
    """Check that the predict_view actually calls the predict system."""
    fake_feed = Mock()
    mock_parser.return_value = fake_feed, None
    mock_predict.return_value = FAKE_PREDICTION

    _ = client.post(PREDICT_ROUTE)  # parser is mocked: no need to send data

    mock_predict.assert_called_once_with(fake_feed)


@pytest.mark.parametrize(
    ['method', 'body_is_correct', 'data_is_corrupted', 'expected_status'],
    [
        ('GET', None, None, 405),  # Unauthorized HTTP method
        ('POST', False, None, 400),
        ('POST', True, True, 500),
        ('POST', True, False, 200),
    ]
)
@patch('src.serve.app.parse_request_body')
@patch('src.serve.app.predict')
def test_predict_view_handlers(mock_predict, mock_parser, client, method,
                               body_is_correct, data_is_corrupted,
                               expected_status):

    if body_is_correct:
        mock_parser.return_value = Mock(), ''
    else:
        mock_parser.return_value = Mock(), 'Error message'
    if data_is_corrupted:
        mock_predict.side_effect = TypeError  # or any other error
    else:
        mock_predict.return_value = FAKE_PREDICTION

    request_object = getattr(client, method.lower())
    response = request_object(PREDICT_ROUTE)

    assert response.status_code == expected_status
