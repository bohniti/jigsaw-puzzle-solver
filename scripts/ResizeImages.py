import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import plotly.express as px
from PIL import Image
from cv2 import imread, copyMakeBorder, imwrite, BORDER_CONSTANT
from os import listdir, path
from pathlib import Path

base_train_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/raw/hisfrag20/'
base_test_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/raw/hisfrag20_test/'
train_result_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/prepared/resized/hisfrag20/'
test_result_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20//prepared/resized/hisfrag20_test/'


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


train_filenames, test_filenames = get_filenames(base_train_path, base_test_path)
train_size, test_size = get_imgsize(train_filenames, test_filenames, base_train_path, base_test_path)
train_size = np.array(train_size)
test_size = np.array(test_size)

train_df = pd.DataFrame(train_size, columns=['x', 'y'])
test_df = pd.DataFrame(test_size, columns=['x', 'y'])


''':cvar

fig = px.histogram(train_df, x="x", title='Training Data Size x-Axis')
fig.show()

fig = px.box(train_df, x="x", title='Training Data Size x-Axis')
fig.show()

fig = px.histogram(train_df, x="y", title='Training Data Size y-Axis')
fig.show()

fig = px.box(train_df, x="y", title='Training Data Size y-Axis')
fig.show()

fig = px.histogram(test_df, x="x", title='Test Data Size x-Axis')
fig.show()

fig = px.box(test_df, x="x", title='Test Data Size x-Axis')
fig.show()

fig = px.histogram(test_df, x="y", title='Test Data Size y-Axis')
fig.show()

fig = px.box(test_df, x="y", title='Test Data Size y-Axis')
fig.show()

'''
print('start')

train_df['filenames'] = train_filenames
test_df['filenames'] = test_filenames

crop_train_df = train_df.copy()
crop_test_df = test_df.copy()

pad_train_df = train_df.copy()
pad_test_df = test_df.copy()

quantiles_values = [0.25, 0.75]
x_quantiles = train_df.x.quantile(quantiles_values)
y_quantiles = train_df.y.quantile(quantiles_values)

pad_train_df = pad_train_df[(pad_train_df.x >= x_quantiles[0.25]) & (pad_train_df.x <= x_quantiles[0.75])]
pad_train_df = pad_train_df[(pad_train_df.y >= y_quantiles[0.25]) & (pad_train_df.y <= y_quantiles[0.75])]

print(pad_train_df.shape)

crop_train_df = crop_train_df[(crop_train_df.x < x_quantiles[0.25]) | (crop_train_df.x > x_quantiles[0.75])]
crop_train_df = crop_train_df[(crop_train_df.y < y_quantiles[0.25]) | (crop_train_df.y > y_quantiles[0.75])]

x_quantiles = test_df.x.quantile(quantiles_values)
y_quantiles = test_df.y.quantile(quantiles_values)

pad_test_df = pad_test_df[(pad_test_df.x >= x_quantiles[0.25]) & (pad_test_df.x <= x_quantiles[0.75])]
pad_test_df = pad_test_df[(pad_test_df.y >= y_quantiles[0.25]) & (pad_test_df.y <= y_quantiles[0.75])]

print(pad_train_df.shape)

crop_test_df = crop_test_df[(crop_test_df.x < x_quantiles[0.25]) | (crop_test_df.x > x_quantiles[0.75])]
crop_test_df = crop_test_df[(crop_test_df.y < y_quantiles[0.25]) | (crop_test_df.y > y_quantiles[0.75])]

print(crop_train_df.shape)

is_train_result_path_empty = not any(Path(train_result_path).iterdir())
is_test_result_path_empty = not any(Path(test_result_path).iterdir())

black = [0, 0, 0]

to_crop_images = [imread(base_train_path + filename + '.jpg') for file in crop_train_df.filenames]
to_crop_test_images = [imread(base_train_path + filename + '.jpg') for file in crop_test_df.filenames]

to_pad_images = [imread(base_train_path + filename + '.jpg') for filename in pad_train_df.filenames]
to_pad_test_images = [imread(base_test_path + filename + '.jpg') for filename in pad_test_df.filenames]

top = int(x_quantiles[0.75])
bottom = int(x_quantiles[0.75])
left = int(y_quantiles[0.75])
right = int(y_quantiles[0.75])

print(top)
print(bottom)
print(left)
print(right)

paded_images = [
    copyMakeBorder(src=file, top=top, bottom=bottom, left=left, right=right, borderType=BORDER_CONSTANT, value=black)
    for file in to_pad_images]

paded_test_images = [
    copyMakeBorder(src=file, top=top, bottom=bottom, left=left, right=right, borderType=BORDER_CONSTANT, value=black)
    for file in to_pad_test_images]

coped_images = [
    copyMakeBorder(src=file, top=top, bottom=bottom, left=left, right=right, borderType=BORDER_CONSTANT, value=black)
    for file in to_crop_images]

coped_test_images = [
    copyMakeBorder(src=file, top=top, bottom=bottom, left=left, right=right, borderType=BORDER_CONSTANT, value=black)
    for file in to_crop_test_images]

for i, img in enumerate(paded_images):
    filename = pad_train_df.filenames.iloc[i]
    imwrite(train_result_path + filename + '.jpg', img)
    break

for i, img in enumerate(paded__test_images):
    filename = pad_train_df.filenames.iloc[i]
    imwrite(train_result_path + filename + '.jpg', img)
    break
