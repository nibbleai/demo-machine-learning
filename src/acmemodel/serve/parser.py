from collections import OrderedDict
from typing import Tuple, List, Optional

from werkzeug.wrappers import Request

# Error messages
MISSING_DATA_KEY = "Request's body must be a JSON object with a 'data' key."
ILL_FORMED_OBJECT = ("'data' must be a single object or an array of objects. "
                     "Each object must represent a valid data point.")

EXPECTED_KEYS = [
    'rally', 'serve', 'hitpoint', 'speed', 'net.clearance',
    'distance.from.sideline', 'depth', 'outside.sideline',
    'outside.baseline', 'player.distance.travelled', 'player.impact.depth',
    'player.impact.distance.from.center', 'player.depth',
    'player.distance.from.center', 'previous.speed',
    'previous.net.clearance', 'previous.distance.from.sideline',
    'previous.depth', 'opponent.depth', 'opponent.distance.from.center',
    'same.side', 'previous.hitpoint', 'previous.time.to.net',
    'server.is.impact.player', 'id'
]


def parse_request_body(request: Request) -> Tuple[Optional[List[dict]], str]:
    """Check request's body and return a sanitized version of payload."""
    try:
        data = request.get_json()['data']
    except (AttributeError, KeyError, TypeError):
        return None, MISSING_DATA_KEY
    else:
        if isinstance(data, dict):  # single data point
            data = [data]

    if (not isinstance(data, list)
        or not all(_is_valid_datapoint(dp) for dp in data)):
        return None, ILL_FORMED_OBJECT

    return [_reorder(datapoint) for datapoint in data], ''


def _is_valid_datapoint(obj: dict) -> bool:
    """Check that all required key/value pairs are set."""
    keys, values = obj.keys(), obj.values()
    return set(keys) == set(EXPECTED_KEYS) and not any(v is None for v in values)


def _reorder(datapoint: dict) -> dict:
    # Ensure that the dictionnary has its keys in the same order than what
    # the prediction pipeline expects.
    return OrderedDict([(key, datapoint[key]) for key in EXPECTED_KEYS])
