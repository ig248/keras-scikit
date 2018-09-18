import tempfile

import numpy as np
import pytest
from keras.layers.core import Activation, Dense
from keras.losses import hinge
from keras.models import Sequential
from keras.utils.test_utils import get_test_data
from sklearn.preprocessing import OneHotEncoder

from scikit_keras.keras.batches import ArrayBatchGenerator
from scikit_keras.keras.wrapper import BaseWrapper, KerasClassifier, KerasRegressor
from scikit_keras.utils.pickle import dumpf, loadf

input_dim = 5
hidden_dims = 5
num_train = 100
num_test = 50
num_classes_binary = 2
num_classes_multi = 3
batch_size = 32
epochs = 1
verbosity = 0
batch_size = 4

np.random.seed(42)


def test_base_class_illegal_param():
    with pytest.raises(ValueError):
        BaseWrapper(param='value')


def test_base_class_not_implemented_model():
    with pytest.raises(NotImplementedError):
        base = BaseWrapper()
        base.model = base.__model__()


def test_base_class_get_set_params():
    params = dict(epochs=66, batch_size=1024)
    base = BaseWrapper(**params)
    assert base.get_params() == params
    params2 = dict(epochs=42, batch_size=128)
    base.set_params(**params2)
    assert base.get_params() == params2


def data_for_classification(num_classes=num_classes_multi, one_hot=False):
    (x_train, y_train), (x_test, y_test) = get_test_data(
        num_train=num_train,
        num_test=num_test,
        input_shape=(input_dim, ),
        classification=True,
        num_classes=num_classes
    )
    if one_hot:
        ohe = OneHotEncoder(sparse=False)
        y_train = ohe.fit_transform(y_train.reshape(-1, 1))
        y_test = ohe.transform(y_test.reshape(-1, 1))

    return (x_train, y_train), (x_test, y_test)


def data_for_regression():
    (x_train, y_train), (x_test, y_test) = get_test_data(
        num_train=num_train,
        num_test=num_test,
        input_shape=(input_dim, ),
        classification=True,
        num_classes=num_classes_multi
    )
    return (x_train, y_train), (x_test, y_test)


class BinaryClassifierSubClass(KerasClassifier):
    def __model__(
        self,
        hidden_dims=hidden_dims,
        num_classes=num_classes_binary,
        loss='binary_crossentropy',
        metrics=None
    ):
        assert num_classes == num_classes_binary
        model = Sequential()
        model.add(Dense(input_dim, input_shape=(input_dim, )))
        model.add(Activation('relu'))
        model.add(Dense(hidden_dims))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(optimizer='sgd', loss=loss, metrics=metrics)
        return model


class SoftmaxClassifierSubClass(KerasClassifier):
    def __model__(
        self,
        hidden_dims=hidden_dims,
        num_classes=num_classes_binary,
        loss='categorical_crossentropy',
        metrics=None
    ):
        model = Sequential()
        model.add(Dense(input_dim, input_shape=(input_dim, )))
        model.add(Activation('relu'))
        model.add(Dense(hidden_dims))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        model.compile(optimizer='sgd', loss=loss, metrics=metrics)
        return model


class RegressionSubClass(KerasRegressor):
    def __model__(self, hidden_dims=hidden_dims, metrics=None):
        model = Sequential()
        model.add(Dense(input_dim, input_shape=(input_dim, )))
        model.add(Activation('relu'))
        model.add(Dense(hidden_dims))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(
            optimizer='sgd', loss='mean_absolute_error', metrics=metrics
        )
        return model


def assert_classification_works(clf, data, generator=False):
    (x_train, y_train), (x_test, y_test) = data
    if len(y_train.shape) == 2 and y_train.shape[1] > 1:
        num_classes = y_train.shape[1]
    else:
        num_classes = len(np.unique(y_train))
    if generator:
        gen_train = ArrayBatchGenerator(
            x_train, y_train, batch_size=batch_size
        )
        clf.fit_generator(gen_train, epochs=epochs, verbose=verbosity)
    else:
        clf.fit(
            x_train,
            y_train,
            sample_weight=np.ones(x_train.shape[0]),
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbosity
        )
    score = clf.score(x_train, y_train, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)
    preds = clf.predict(x_test, batch_size=batch_size)
    assert preds.shape == (num_test, )
    for prediction in np.unique(preds):
        assert prediction in range(num_classes)
    proba = clf.predict_proba(x_test, batch_size=batch_size)
    assert proba.shape == (num_test, num_classes)
    assert np.allclose(np.sum(proba, axis=1), np.ones(num_test))


def assert_string_classification_works(clf, data):
    (x_train, y_train), (x_test, y_test) = data
    assert len(y_train.shape) == 1
    num_classes = len(np.unique(y_train))
    string_classes = ['cls{}'.format(x) for x in range(num_classes)]
    str_y_train = np.array(string_classes)[y_train]

    clf.fit(x_train, str_y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbosity)

    score = clf.score(x_train, str_y_train, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)

    preds = clf.predict(x_test, batch_size=batch_size)
    assert preds.shape == (num_test, )
    for prediction in np.unique(preds):
        assert prediction in string_classes

    proba = clf.predict_proba(x_test, batch_size=batch_size)
    assert proba.shape == (num_test, num_classes)
    assert np.allclose(np.sum(proba, axis=1), np.ones(num_test))


def assert_regression_works(reg, data, generator=False):
    (x_train, y_train), (x_test, y_test) = data

    if generator:
        gen_train = ArrayBatchGenerator(
            x_train, y_train, batch_size=batch_size
        )
        reg.fit_generator(gen_train, epochs=epochs, verbose=verbosity)
    else:
        reg.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbosity)

    score = reg.score(x_train, y_train, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)

    preds = reg.predict(x_test, batch_size=batch_size)
    assert preds.shape == (num_test, )


def assert_models_equal(first, second):
    """two keras Model instances are equal enough"""
    # layer names and settings
    assert first.get_config() == second.get_config()
    # model weights
    assert len(first.get_weights()) == len(second.get_weights())
    for w1, w2 in zip(first.get_weights(), second.get_weights()):
        np.testing.assert_array_equal(w1, w2)
    # optimizer
    assert first.optimizer.get_config() == second.optimizer.get_config()


def assert_wrappers_equal(first, second):
    """two BaseWrapper instances are equal enough"""
    assert first.sk_params == second.sk_params
    assert first.history_ == second.history_
    if not first.model_ or not second.model_:
        assert first.model_ == second.model_
    else:
        assert_models_equal(first.model, second.model)


def assert_predictions_equal(first, second, x):
    """two BaseWrapper instances return same predictions"""
    preds1 = first.predict(x, batch_size=batch_size)
    preds2 = second.predict(x, batch_size=batch_size)
    np.testing.assert_array_equal(preds1, preds2)


def assert_pickling_works(wrpr, data):
    x = data[0][0]  # x_train
    with tempfile.NamedTemporaryFile(suffix='.pickle', delete=True) as f:
        dumpf(wrpr, f.name)
        wrpr2 = loadf(f.name)
    assert_wrappers_equal(wrpr, wrpr2)
    if wrpr.model_ is not None:
        assert_predictions_equal(wrpr, wrpr2, x)


@pytest.mark.parametrize('generator', [True])
@pytest.mark.parametrize(
    'clf_cls, num_classes, one_hot', [
        (BinaryClassifierSubClass, 2, False),
        (SoftmaxClassifierSubClass, 2, True),
        (SoftmaxClassifierSubClass, 3, True)
    ]
)
def test_classifiers_generator(clf_cls, num_classes, one_hot, generator):
    clf = clf_cls(
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        metrics=['accuracy'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbosity
    )
    data = data_for_classification(num_classes=num_classes, one_hot=one_hot)
    assert clf.model_ is None
    assert_classification_works(clf, data, generator=generator)


@pytest.mark.parametrize('generator', [False])
@pytest.mark.parametrize(
    'one_hot', [
        False,
        True,
    ], ids=['integer', 'onehot']
)
@pytest.mark.parametrize(
    'num_classes', [
        2,
        3,
    ], ids=['binary', 'multiclass']
)
@pytest.mark.parametrize(
    'clf_cls', [
        BinaryClassifierSubClass,
        SoftmaxClassifierSubClass,
    ],
    ids=['sigmoid', 'softmax']
)
def test_classifiers(clf_cls, num_classes, one_hot, generator):
    if (
        clf_cls is BinaryClassifierSubClass and
            num_classes != num_classes_binary
    ):
        pytest.skip(
            'Invalid: Binary classifier with {} classes'.format(num_classes)
        )
    if clf_cls is BinaryClassifierSubClass and one_hot:
        pytest.skip('Invalid: Binary classifier with one-hot encoding')
    clf = clf_cls(
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        metrics=['accuracy'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbosity
    )
    data = data_for_classification(num_classes=num_classes, one_hot=one_hot)
    assert clf.model_ is None
    assert_pickling_works(clf, data)  # before fitting
    assert_classification_works(clf, data, generator=generator)
    if not one_hot:
        assert_string_classification_works(clf, data)
    assert_pickling_works(clf, data)  # after fitting


@pytest.mark.parametrize('generator', [False, True])
@pytest.mark.parametrize(
    'loss',
    [
        'hinge',
        hinge  # loss as function
    ],
    ids=['string', 'function']
)
def test_classifier_losses(loss, generator):
    num_classes = num_classes_binary
    clf = BinaryClassifierSubClass(
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        loss=loss,
        metrics=['accuracy'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbosity
    )
    data = data_for_classification(num_classes=num_classes)
    assert_classification_works(clf, data, generator=generator)


@pytest.mark.parametrize('generator', [False, True])
@pytest.mark.parametrize(
    'metrics', [None, ['accuracy']],
    ids=['no_extra_metrics', 'with_extra_metrics']
)
def test_regression_subclass(metrics, generator):
    reg = RegressionSubClass(
        hidden_dims=hidden_dims,
        metrics=metrics,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbosity
    )
    data = data_for_regression()
    assert reg.model_ is None
    assert_pickling_works(reg, data)  # before fitting
    assert_regression_works(reg, data, generator=generator)
    assert_pickling_works(reg, data)  # after fitting


@pytest.mark.parametrize(
    'num_classes', [
        2,
    ], ids=['binary']
)
@pytest.mark.parametrize(
    'clf_cls', [
        BinaryClassifierSubClass,
        SoftmaxClassifierSubClass,
    ],
    ids=['sigmoid', 'softmax']
)
def test_classifier_no_compiled_accuracy(clf_cls, num_classes):
    clf = clf_cls(
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        metrics=None,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbosity
    )
    data = data_for_classification(num_classes=num_classes)
    with pytest.raises(ValueError):
        assert_classification_works(clf, data)


def test_invalid_y_shape():
    clf = BinaryClassifierSubClass(batch_size=batch_size, epochs=epochs)
    y = np.ones((3, 4, 5))
    with pytest.raises(ValueError):
        clf.fit(y, y)


if __name__ == '__main__':
    pytest.main()
