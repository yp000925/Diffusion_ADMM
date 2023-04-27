
import argparse


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def prop_params():
    res = dict(
        w = 632e-9,
        deltax = 3.45e-6,
        deltay = 3.45e-6,
        distance = 0.02,
        nx = 512,
        ny = 512
    )
    return dict2namespace(res)