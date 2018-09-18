import tempfile
from typing import Dict, Iterable, Union

from keras.engine import Layer
from keras.models import load_model, save_model


def iterlayers(model: Layer) -> Iterable[Layer]:
    """
    Return iterable over all layers (and sub-layers) of a model.

    This works because keras Model is a sub-class of Layer.
    Can be used for freezing/un-freezing layers, etc.
    """
    if hasattr(model, 'layers'):
        for layer in model.layers:
            yield from iterlayers(layer)
    else:
        yield model


def fix_model_duplicate_layers_by_name(
    model: Layer, substitutions=None, return_substitutions=False
) -> Layer:
    """Fix duplicates in loaded model by name."""
    if not substitutions:
        substitutions = {}
    if hasattr(model, 'layers'):
        for idx, layer in enumerate(model.layers):
            if layer.name in substitutions:
                model.layers[idx] = substitutions[layer.name]
            else:
                model.layers[idx] = fix_model_duplicate_layers_by_name(
                    layer, substitutions=substitutions
                )
                substitutions[layer.name] = layer
    if return_substitutions:
        return model, substitutions
    else:
        return model


def fix_multimodel_duplicate_layers_by_name(
    multimodel: Dict[str, Layer],
    substitutions=None,
    return_substitutions=False
) -> Layer:
    """Fix duplicates in multi-model by layer/submodel name."""
    if not substitutions:
        substitutions = {}
    for model_name, model in multimodel.items():
        multimodel[model_name], substitutions = (
            fix_model_duplicate_layers_by_name(
                model, substitutions=substitutions, return_substitutions=True
            )
        )
    if return_substitutions:
        return multimodel, substitutions
    else:
        return multimodel


def serialize_model(model: Union[None, Layer]) -> Union[None, bytes]:
    if not model:
        return model
    else:
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            save_model(model, fd.name, overwrite=True)
            model_str = fd.read()
        return model_str


def deserialize_model(
        model_str: Union[None, bytes],
        fix_duplicates=True
) -> Union[None, Layer]:
    if not model_str:
        return model_str
    else:
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(model_str)
            fd.flush()
            model = load_model(fd.name)
        if fix_duplicates:
            model = fix_model_duplicate_layers_by_name(model)
        return model
