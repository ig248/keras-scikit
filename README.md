[![Build Status](https://travis-ci.com/ig248/scikit-keras.svg?branch=master)](https://travis-ci.com/ig248/scikit-keras)
# scikit-keras
Scikit-like keras wrapper with support for callbacks and temporal tasks.
Base on [Keras](https://keras.io/scikit-learn-api/), but with additional features.

## Installation
`pip install .` or `make install`

## Usage
Subclass `KerasClassifier` or `KerasRegressor`,
and implement `.__model__()` and `.__calbacks__()`
