import os
import random as rn

import numpy as np
import pytest
import tensorflow as tf
from keras import backend as K  # noqa


@pytest.fixture
def random():
    """Seed all random number generators.

    Ensures repeatable tests
    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
