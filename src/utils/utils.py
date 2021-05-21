import pickle


def save_obj(obj, name):
    with open('../config/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('../config/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
