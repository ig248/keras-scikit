from joblib.externals.cloudpickle import dump, load
from joblib.externals.cloudpickle import dumps, loads  # noqa

__all__ = ['dumpf', 'loadf']


def dumpf(obj, filename):
    """Serialize object to file of given name."""
    with open(filename, 'wb') as file:
        dump(obj, file)


def loadf(filename):
    """Load serialized object from file of given name."""
    with open(filename, 'rb') as file:
        obj = load(file)
    return obj
