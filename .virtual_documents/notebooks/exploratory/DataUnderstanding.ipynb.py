# import packages
from os import listdir, path
from os.path import isfile, join, splitext
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


# declare global variables 
train = "/Users/beantown/Projekte/Jigsaw-Puzzling/Data/hisfrag20"
test = "/Users/beantown/Projekte/Jigsaw-Puzzling/Data/hisfrag20_test"


# prepare environtment
get_ipython().run_line_magic("matplotlib", " inline")


# prepare plots

from helper_functions import set_size
sns.set(style="whitegrid")
plt.style.use('seaborn')
width = 496.85625


def get_info(data_path):
    # load file names train dataset
    file_names = [splitext(f)[0] for f in listdir(data_path) if isfile(join(data_path, f))]

    # load file name test dataset
    #file_names_test = [splitext(f)[0] for f in listdir(data_path_test) if isfile(join(data_path_test, f))]

    # Split the image naming in wirter, page and fragment
    # For training
    #file_names_parts = [i.split("_") for i in file_names]
    # For test
    file_names_parts = [i.split("_") for i in file_names_test]

    return pd.DataFrame.from_records(file_names_parts,columns=['writer_id', 'page_id','fragment_id'])

df get_info(data_path=test)





df.head()
