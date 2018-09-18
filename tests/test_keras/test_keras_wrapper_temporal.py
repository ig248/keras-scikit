import numpy as np
import pytest
from keras.layers import Conv1D, Dense, Input, TimeDistributed
from keras.models import Model

from scikit_keras.keras.wrapper import KerasClassifier

np.random.seed(42)


class SequenceBinaryClassifier(KerasClassifier):
    """Simple binary classifier model with stride 2"""

    def __model__(
        self, n_channels=2, kernel_size=3, strides=2, padding='valid'
    ):
        input_ = Input(shape=(None, n_channels))
        x = Conv1D(
            filters=2,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding
        )(input_)
        output = TimeDistributed(Dense(1, activation='sigmoid'))(x)
        model = Model(input_, output)
        model.compile(optimizer='sgd', loss='binary_crossentropy')
        return model


@pytest.mark.parametrize(
    'n_timesteps, kernel_size, stride, padding, n_labels_expected',
    [
        # (3, 1, 'valid', 1),  # fails because of squeeze
        (4, 3, 1, 'valid', 2),
        (4, 3, 1, 'same', 4),
        (4, 3, 1, 'causal', 4),
        (40, 5, 1, 'causal', 40),
    ]
)
@pytest.mark.parametrize('n_channels', [1, 3])
def test_sequence_binary_classifier_predict_shape(
    n_timesteps, kernel_size, stride, padding, n_labels_expected, n_channels
):
    n_samples = 10

    seqclf = SequenceBinaryClassifier(
        n_channels=n_channels,
        strides=stride,
        padding=padding,
        kernel_size=kernel_size
    )

    seqclf.classes_ = np.arange(2)

    x = np.random.randn(*(n_samples, n_timesteps, n_channels))
    y_pred = seqclf.predict(x)
    y_proba = seqclf.predict_proba(x)
    assert y_pred.shape == (n_samples, n_labels_expected)
    assert y_proba.shape == (n_samples, n_labels_expected, 2)
