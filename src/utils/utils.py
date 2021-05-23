import toml
from ray import tune


def load_config(name, show=True, tune_config=True):
    file = toml.load('../config/' + name + '.toml')
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
