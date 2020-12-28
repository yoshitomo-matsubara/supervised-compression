COMPRESSION_MODEL_CLASS_DICT = dict()
COMPRESSION_MODEL_FUNC_DICT = dict()


def register_compression_model_class(cls):
    COMPRESSION_MODEL_CLASS_DICT[cls.__name__] = cls
    return cls


def register_compression_model_func(func):
    COMPRESSION_MODEL_FUNC_DICT[func.__name__] = func
    return func


def get_compression_model(model_name, **kwargs):
    if model_name in COMPRESSION_MODEL_CLASS_DICT:
        return COMPRESSION_MODEL_CLASS_DICT[model_name](**kwargs)
    elif model_name in COMPRESSION_MODEL_FUNC_DICT:
        return COMPRESSION_MODEL_FUNC_DICT[model_name](**kwargs)
    raise ValueError('model_name `{}` is not expected'.format(model_name))
