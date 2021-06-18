# TODO Change code completly to pathlib for crossfunctionality
from os import listdir, path
from PIL import Image
from pathlib import Path
from cv2 import imread, copyMakeBorder, imwrite, BORDER_CONSTANT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


import plotly.graph_objects as go
fig = go.FigureWidget(data=go.Bar(y=[2, 3, 1]))
fig


train_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/raw/hisfrag20/'
test_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/raw/hisfrag20_test/'
train_result_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/prepared/resized/hisfrag20/'
test_result_path = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20//prepared/resized/hisfrag20_test/'


def get_filenames(train_path, test_path):
    # load file names train dataset
    train = [path.splitext(f)[0] for f in listdir(train_path) if path.isfile(path.join(train_path, f))]

    # load file name test dataset
    test = [path.splitext(f)[0] for f in listdir(test_path) if path.isfile(path.join(test_path, f))]
    
    return train, test


def get_imgsize(train_filenames, test_filenames, train_path, test_path):
    train = [Image.open(train_path + f + '.jpg', 'r').size for f in train_filenames]
    test = [Image.open(test_path + f + '.jpg', 'r').size for f in test_filenames]
    return train, test


train_filenames, test_filenames = get_filenames(train_path, test_path)


train_size, test_size = get_imgsize(train_filenames, test_filenames, train_path, test_path)


train_size = np.array(train_size)
test_size = np.array(test_size)


train_size.shape


test_size.shape


train_df = pd.DataFrame(train_size, columns=['x','y'])
test_df = pd.DataFrame(test_size, columns=['x','y'])


train_df.shape


test_df.shape


train_df.head()


train_df.x.mean()


train_df.y.mean()


test_df.head()


df = pd.concat([test_df, train_df])


df.head()


df.shape


df.x.mean()


df.y.mean()


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


train_df['filenames'] = train_filenames
test_df['filenames'] = test_filenames


train_df.shape


test_df.shape


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

print(crop_train_df.shape)


x_quantiles = test_df.x.quantile(quantiles_values)
y_quantiles = test_df.y.quantile(quantiles_values)

pad_test_df = pad_test_df[(pad_test_df.x >= x_quantiles[0.25]) & (pad_test_df.x <= x_quantiles[0.75])]
pad_test_df = pad_test_df[(pad_test_df.y >= y_quantiles[0.25]) & (pad_test_df.y <= y_quantiles[0.75])]

print(pad_test_df.shape)

crop_test_df = crop_test_df[(crop_test_df.x < x_quantiles[0.25]) | (crop_test_df.x > x_quantiles[0.75])]
crop_test_df = crop_test_df[(crop_test_df.y < y_quantiles[0.25]) | (crop_test_df.y > y_quantiles[0.75])]

print(crop_test_df.shape)


is_train_result_path_empty = not any(Path(train_result_path).iterdir())
is_test_result_path_empty = not any(Path(test_result_path).iterdir())


pad_train_df.head()


pad_train_df.head()


black = [0,0,0]

#to_crop_images = [imread(file) for file in crop_train_df.filenames]
to_pad_images = [imread(train_path + filename + '.jpg') for filename in pad_train_df.filenames]
plt.imshow(to_pad_images[0])


top = int(x_quantiles[0.75])
bottom = int(x_quantiles[0.75])
left = int(y_quantiles[0.75])
right= int(y_quantiles[0.75])

print(top)
print(bottom)
print(left)
print(right)

paded_images = [copyMakeBorder(src=file, top=top, bottom=bottom, left=left, right=right, borderType=BORDER_CONSTANT,value=black) for file in to_pad_images]

for i, img in enumerate(paded_images):
    filename = pad_train_df.filenames.iloc[i]
    imwrite(train_result_path + filename + '.jpg', img)






#if is_train_result_path_empty:
        
    
    
    '''
    print(f'Train images not found in train_result_path.\nStart resizing and save results in: \n{train_result_path}\n')
    for filename in train_filenames:
        src = imread(train_path + filename + '.jpg')

        img = copyMakeBorder(src=src, top=test_size[0], bottom=test_size[0], left=test_size[1], right=test_size[1],borderType=BORDER_CONSTANT,value=black)
        imwrite(train_result_path + filename + '.jpg', img)
    '''
    



else:
    print(f'Resized train images found in: \n{train_result_path}\nSkip resizing for train imgs\n')

if is_test_result_path_empty:

    print(f'Test images found in test_result_path.\nStart resizing and save results in: \n{test_result_path}\n')

    for filename in test_filenames:
        src = imread(test_path + filename + '.jpg')
        img = copyMakeBorder(src=src, top=test_size[0], bottom=test_size[0], left=test_size[1], right=test_size[1],borderType=BORDER_CONSTANT,value=black)
        imwrite(test_result_path + filename + '.jpg', img)
else:
    print(f'Resized test images found in: \n{test_result_path}\nSkip resizing for test imgs\n')



