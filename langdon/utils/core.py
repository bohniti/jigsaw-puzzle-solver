from sys import platform

import toml


def load_config(name, show=True, tune_config=True):
    # TODO make this work in a clean way for every domin (docker, mac, win ...)
    try:
        file = toml.load(name)

    except:
        file = toml.load(name)

    if tune_config:
        config = {}

        for param, spec in file.items():
            type_ = spec.pop("type")
            config[param] = getattr(tune, type_)(**spec)
        if show:
            show_config(file, config_type='Tune Search Space')
        return config

    if show:
        show_config(file, config_type='General')
    return file


def show_config(file, config_type):
    # print(f'#######################\n# {config_type} CONFIGURATION #\n#######################')
    print('{:s}'.format('\u0332'.join(f'{config_type} Configuration:')))
    for key, value in file.items():
        print(f'{key}: {value} ')
    print('')


def get_config_path():
    if platform == "linux" or platform == "linux2":
        config = '/home/hpc/iwi5/iwi5012h/dev/jigsaw-puzzle-solver/config/config.toml'
    elif platform == "darwin":
        config = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/config/config_local.toml'
    elif platform == "win32":
        raise NotImplementedError
    return config
