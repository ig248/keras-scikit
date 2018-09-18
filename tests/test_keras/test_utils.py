import pytest
import keras.backend as K
from keras.layers import Concatenate, Dense, Input
from keras.models import Model, Sequential

from scikit_keras.keras.utils import (iterlayers,
                                      deserialize_model,
                                      serialize_model)


def model_single_layer():
    model = Dense(10)
    return model


def model_sequential():
    model = Sequential()
    model.add(Dense(10))
    model.add(Dense(20))
    return model


def model_compiled():
    model = Sequential()
    model.add(Dense(10))
    model.add(Dense(20))
    model.compile(optimizer='sgd', loss='mse')
    return model


def model_with_shared_layer():
    x1 = Input((10, ))
    x2 = Input((10, ))
    dense = Dense(1)
    concat = Concatenate()([dense(x1), dense(x2)])
    model = Model([x1, x2], concat)
    return model


def model_with_submodel():
    submodel = Sequential()
    submodel.add(Dense(10))
    submodel.add(Dense(20))
    model = Sequential()
    model.add(submodel)
    model.add(Dense(30))
    return model


def model_with_shared_in_submodel():
    x = Input((10, ))
    dense = Dense(10)
    submodel = Model(x, dense(x))
    model = Model(x, dense(submodel(x)))
    return model


class TestIterLayers:
    """Test iteration over complex model layers"""

    @pytest.mark.parametrize(
        'model, n_layers',
        [
            (None, 1),  # Let's see if that's what we want
            (model_single_layer(), 1),
            (model_sequential(), 2),
            (model_compiled(), 2),
            (model_with_submodel(), 3)
        ]
    )
    def test_model(self, model, n_layers):
        layers = list(iterlayers(model))
        assert len(layers) == n_layers


class TestSerializeModel:
    """Test serializing and deserializing a single Keras model"""

    def test_none(self):
        model = None
        model_str = serialize_model(model)
        assert model_str is None
        model2 = deserialize_model(model_str)
        assert model2 is None

    @pytest.mark.parametrize(
        'model_func', [
            model_sequential,
            model_compiled,
            model_with_submodel,
            model_with_shared_layer,
            model_with_shared_in_submodel,
        ]
    )
    def test_model(self, model_func):
        model = model_func()
        model_str = serialize_model(model)
        assert isinstance(model_str, bytes)
        model2 = deserialize_model(model_str)
        assert len(model.layers) == len(model2.layers)  # shallow comparison
        assert len(set(model.layers)) == len(set(model2.layers)
                                             )  # check num of shared layers
        layers = list(iterlayers(model))
        layers2 = list(iterlayers(model2))
        assert len(layers) == len(layers2)  # deep comparison
        assert len(set(layers)) == len(set(layers2)
                                       )  # check num of shared layers

    @pytest.mark.parametrize(
        'model_func', [
            model_with_shared_layer,
            model_with_shared_in_submodel,
        ]
    )
    def test_model_gradients(self, model_func):
        model = model_func()
        model.compile(loss='mse', optimizer='sgd')
        model_str = serialize_model(model)
        model = deserialize_model(model_str)
        grads = K.gradients(model.total_loss, model.trainable_weights)
        assert all([g is not None for g in grads])
