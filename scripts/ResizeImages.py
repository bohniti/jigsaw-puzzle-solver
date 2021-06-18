# File not be used anymore!!!

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from cv2 import imread, copyMakeBorder, imwrite, BORDER_CONSTANT
from os import listdir, path
from pathlib import Path

base_train_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/raw/hisfrag20/'
base_test_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/raw/hisfrag20_test/'
train_result_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/prepared/resized/hisfrag20/'
test_result_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20//prepared/resized/hisfrag20_test/'
quantiles_values = [0.25, 0.75]
black = [0, 0, 0]

def get_filenames(train_path, test_path):
    # load file names train dataset
    train = [path.splitext(f)[0] for f in listdir(train_path) if path.isfile(path.join(train_path, f))]
    # load file name test dataset
    test = [path.splitext(f)[0] for f in listdir(test_path) if path.isfile(path.join(test_path, f))]
    return train, test


def get_imgsize(trainFilenames, testFilenames, train_path, test_path):
    train = [Image.open(train_path + f + '.jpg', 'r').size for f in trainFilenames]
    test = [Image.open(test_path + f + '.jpg', 'r').size for f in testFilenames]
    return train, test

print('Start with getting file size')
train_filenames, test_filenames = get_filenames(base_train_path, base_test_path)
train_size, test_size = get_imgsize(train_filenames, test_filenames, base_train_path, base_test_path)
train_size = np.array(train_size)
test_size = np.array(test_size)

train_df = pd.DataFrame(train_size, columns=['x', 'y'])
test_df = pd.DataFrame(test_size, columns=['x', 'y'])

print('start calculating sizes and subset order')

train_df['filenames'] = train_filenames
test_df['filenames'] = test_filenames


# Get the size by which we split the data
x_quantiles = train_df.x.quantile(quantiles_values)
y_quantiles = train_df.y.quantile(quantiles_values)
x_test_quantiles = test_df.x.quantile(quantiles_values)
y_test_quantiles = test_df.y.quantile(quantiles_values)

# Get size for cropping and padding
top = int(x_quantiles[0.75])
bottom = int(x_quantiles[0.75])
left = int(y_quantiles[0.75])
right = int(y_quantiles[0.75])


# Split the dataframes such that we crop and pad some img instead of pad everything
#pad_train_df = train_df.copy()
#pad_train_df = pad_train_df[(pad_train_df.x >= x_quantiles[0.25]) & (pad_train_df.x <= x_quantiles[0.75])]

pad_test_df = test_df.copy()
pad_test_df = pad_test_df[(pad_test_df.x >= x_test_quantiles[0.25]) & (pad_test_df.x <= x_test_quantiles[0.75])]

crop_train_df = train_df.copy()
crop_train_df = crop_train_df[(crop_train_df.x < x_quantiles[0.25]) | (crop_train_df.x > x_quantiles[0.75])]

crop_test_df = test_df.copy()
crop_test_df = crop_test_df[(crop_test_df.x < x_test_quantiles[0.25]) | (crop_test_df.x > x_test_quantiles[0.75])]



print('Start reading pad')
# pad imgs which are defined above and in the train dataset
# read pad train
images = [imread(base_train_path + filename + '.jpg') for filename in pad_train_df.filenames]

print('pading train imgs & writing pad train imgs')
# pad train & write train pad
for i, img in enumerate(images):
    img = copyMakeBorder(src=img, top=top, bottom=bottom, left=left, right=right, borderType=BORDER_CONSTANT, value=black)
    filename = pad_train_df.filenames.iloc[i]
    imwrite(train_result_path + filename + '.jpg', img)

print('Start reading pad test')
# read pad test
images = [imread(base_test_path + filename + '.jpg') for filename in pad_test_df.filenames]

print('Start pading test imgs & writing pad test imgs')
for i, img in enumerate(images):
    img = copyMakeBorder(src=img, top=top, bottom=bottom, left=left, right=right, borderType=BORDER_CONSTANT, value=black)
    filename = pad_test_df.filenames.iloc[i]
    imwrite(test_result_path + filename + '.jpg', img)

# crop imgs which are defined above and in the train dataset
# read crop train
print('Start reading crop')
images = [imread(base_train_path + filename + '.jpg') for filename in crop_train_df.filenames]

# crop train
print('Start cropping train imgs')
images = [file[0:0+left, 0:0+top] for file in images]

# write crop train
print('Start writing crop train imgs')
for i, img in enumerate(images):
    filename = crop_train_df.filenames.iloc[i]
    imwrite(train_result_path + filename + '.jpg', img)



# crop imgs which are defined above and in the test dataset
# read crop test
print('Start reading crop test')
images = [imread(base_test_path + filename + '.jpg') for filename in crop_test_df.filenames]
# crop test
print('Start cropping test imgs')
images = [file[0:0+left, 0:0+top] for file in images]

# write crop test
print('Start writing crop test imgs')
for i, img in enumerate(images):
    filename = crop_test_df.filenames.iloc[i]
    imwrite(test_result_path + filename + '.jpg', img)

