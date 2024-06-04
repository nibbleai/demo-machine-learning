"""Tests for AWS wrappers.

NOTE
====

This way of testing can be seen as a gentle mix between unit and integration:
    - we stub boto3 behavior
    - we check that the wrapper is actually calling boto3 by checking the stub
    afterwards


Another solution, maybe more "atomic", would be to:
    - patch the S3 client from `src.aws`
    - assert that this client is called when calling our function

Both would be valid in our opinion. Remember: more than the label that you put
on your test, what's important is what your test do.

The second implementation  is left as an exercice for the motivated reader ;-)
"""
from unittest.mock import Mock

from botocore.stub import Stubber
import pytest

from src.aws import upload_data_to_s3, download_data_from_s3, S3
from src.config import AWS_S3_BUCKET_NAME

FAKE_DATA = b'Just some random bytes...'
FAKE_KEY = 'folder/filename'


@pytest.fixture
def s3_stub():
    with Stubber(S3) as stub:
        yield stub
        # To make it clearer you could check for no prending responses in
        # each individual functions. Setting it here avoid repetitions though.
        stub.assert_no_pending_responses()


def test_upload_to_s3(s3_stub):
    fake_params = {
        'ACL': 'private',
        'Bucket': AWS_S3_BUCKET_NAME,
        'Body': FAKE_DATA,
        'Key': FAKE_KEY
    }
    # We don't use the object returned by AWS, but only the side-effet of the
    # `put_object` method. An empty dict as fake response is OK
    s3_stub.add_response('put_object', {}, fake_params)

    upload_data_to_s3(FAKE_DATA, FAKE_KEY)


def test_dowmload_from_s3(s3_stub):
    fake_body = Mock()
    fake_body.read.return_value = FAKE_DATA
    fake_params = {
        'Bucket': AWS_S3_BUCKET_NAME,
        'Key': FAKE_KEY
    }
    # Here we need a returned object with a 'Body' key
    s3_stub.add_response('get_object', {'Body': fake_body}, fake_params)

    data = download_data_from_s3(FAKE_KEY)

    assert data == FAKE_DATA
