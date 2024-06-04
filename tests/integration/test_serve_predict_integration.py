import json
import os
from random import random

import pytest

from src.serve.app import app as flask_app, PREDICT_ROUTE
from tests.utils import get_test_datapoints

flask_app.config['TESTING'] = True


@pytest.fixture
def client():
    with flask_app.test_client() as client:
        yield client


@pytest.mark.integration
@pytest.mark.parametrize(
    ['corrupted_data', 'ill_formated_body', 'expected_status'],
    [
        (True, False, 500),
        (False, True, 400),
        (False, False, 200)
])
@pytest.mark.parametrize('data_point_count', [1, 10, 100])
def test_predict_view_integration(data_point_count, corrupted_data,
                                  ill_formated_body, expected_status, client):
    """Check that the predict view can call the prediction system."""
    data_points = get_test_datapoints(data_point_count)
    if corrupted_data:
        # Randomly corrupt somes value by insterting meanless content
        # Note: this could be done in a fixture...
        for dp in data_points:
            for key in dp.keys():
                if random() > 0.5:
                    dp[key] = str(os.urandom(10))

    if ill_formated_body:
        # Just build a feed that does NOT comply to the API documentation.
        # Here we use `payload` as the root key, instead of `data`
        feed = {'payload': data_points}
    else:
        feed = {'data': data_points}

    response = client.post(PREDICT_ROUTE, json=feed)

    _ = json.loads(response.data)  # check valid JSON response
    assert response.status_code == expected_status
