import toml


def load_config(name, show=True):

    file = toml.load('../config/' + name + '.toml')
    if show:
        print(f'#######################\n# MODEL CONFIGURATION #\n#######################')
        for key, value in file.items():
            print(f'{key}: {value} ')
    return file

