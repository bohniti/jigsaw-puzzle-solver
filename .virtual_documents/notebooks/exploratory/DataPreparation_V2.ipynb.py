import pandas as pd
from os import listdir
from os.path import isfile, join, splitext
from sklearn.model_selection import GroupShuffleSplit
import random
import csv


train_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/raw/hisfrag20/'
test_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/raw/hisfrag20_test/'
result_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/prepared/paris_as_csv/'


def get_info(data_path):
    '''Helper Method which provids the path onto you local machine and the fragment ID, such that pairs can be created'''
    # load file names train dataset
    file_names = [splitext(f)[0] for f in listdir(data_path) if isfile(join(data_path, f))]

    # load file name test dataset
    #file_names_test = [splitext(f)[0] for f in listdir(data_path_test) if isfile(join(data_path_test, f))]

    # Split the image naming in wirter, page and fragment
    # For training
    #file_names_parts = [i.split("_") for i in file_names]
    # For test
    file_names_parts = [i.split("_") for i in file_names]
    df = pd.DataFrame.from_records(file_names_parts,columns=['writer_id', 'ID_of_original_papyrus','fragment_id'])
    df['path_to_fragment_image'] = file_names
    df['path_to_fragment_image'] = data_path +  df['path_to_fragment_image'].astype(str)
    return df[['path_to_fragment_image', 'ID_of_original_papyrus']]

df = get_info(data_path=train_path)


gs = GroupShuffleSplit(n_splits=2, test_size=.2, random_state=0)
train_idx, val_idx = next(gs.split(df, groups=df.ID_of_original_papyrus))


train = df.loc[train_idx]
val = df.loc[val_idx]


train.ID_of_original_papyrus.isin(val.ID_of_original_papyrus).any()


train.shape


val.shape


test = get_info(data_path=test_path)


train.ID_of_original_papyrus.isin(test.ID_of_original_papyrus).any()


val.ID_of_original_papyrus.isin(test.ID_of_original_papyrus).any()


def sample_pairs(K, data, IDList):
    """
    used from: https://github.com/plnicolas/master-thesis/blob/master/Papy-S-Net/PairGenerator.py
    Function to create fragment pairs given a Pandas DataFrame and a list of IDs.
    Parameters:
    ----------
        - K: The number of pairs of each type (positive and negative) to sample. Duplicates will be dropped,
        so the final number of pairs WILL be smaller than 2K
        - Data: Pandas DataFrame with rows of the form [path_to_fragment_image, ID_of_original_papyrus]
        - IDList: List containing the IDs of the papyri to sample fragments from
    Returns:
    --------
        - pairs: A list of fragment pairs, of the form [path_to_frag1, path_to_frag2]
        - labels: A list of labels, i.e. original papyrus IDs
    """

    pairs = []
    labels = []

    # For each papyrus used for training
    for index in IDList:
        isIndex = data.iloc[:, 1] == index
        isNotIndex = data.iloc[:, 1] get_ipython().getoutput("= index")
        # List of images from the indexed papyrus
        indexTrueList = data[isIndex].iloc[:, 0]
        # List of images NOT from the indexed papyrus
        indexFalseList = data[isNotIndex].iloc[:, 0]

        # K negative pairs
        p1List = indexTrueList.sample(n=K, replace=True, random_state=356)
        p2List = indexFalseList.sample(n=K, replace=True, random_state=323)
        for k in range(K):
            pair = [p1List.values[k], p2List.values[k]]
            if pair not in pairs:
                pairs.append(pair)
                labels.append(1)

        # K positive pairs
        p1List = indexTrueList.sample(n=K, replace=True, random_state=362)
        p2List = indexTrueList.sample(n=K, replace=True, random_state=316)
        for k in range(K):
            pair = [p1List.values[k], p2List.values[k]]
            if pair not in pairs:
                pairs.append(pair)
                labels.append(0)

    # Shuffle the pairs and label lists before returning them
    # The two lists are shuffled at once with the same order, of course
    tmp = list(zip(pairs, labels))
    random.shuffle(tmp)
    pairs, labels = zip(*tmp)

    return pairs, labels


X_train, y_train = sample_pairs(K=2, data=train, IDList=train.ID_of_original_papyrus.unique())
X_val, y_val = sample_pairs(K=2, data=val, IDList=val.ID_of_original_papyrus.unique())
X_test, y_test = sample_pairs(K=2, data=test, IDList=test.ID_of_original_papyrus.unique())


train = pd.DataFrame(X_train)
train['y'] = y_train
train.to_csv(result_path + 'train.csv', index=False)


val = pd.DataFrame(X_val)
val['y'] = y_val
val.to_csv(result_path + 'val.csv', index=False)


test = pd.DataFrame(X_test)
test['y'] = y_test
test.to_csv(result_path + 'test.csv', index=False)



