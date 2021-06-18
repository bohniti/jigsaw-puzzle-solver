# import packages
from os import listdir, path
from os.path import isfile, join, splitext
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


# declare global variables 
data_path = "/Users/beantown/Projekte/Jigsaw-Puzzling/Data/hisfrag20"
data_path_test = "/Users/beantown/Projekte/Jigsaw-Puzzling/Data/hisfrag20_test"


# prepare environtment
get_ipython().run_line_magic("matplotlib", " inline")


# prepare plots

from helper_functions import set_size
sns.set(style="whitegrid")
plt.style.use('seaborn')
width = 496.85625


# load file names train dataset
file_names = [splitext(f)[0] for f in listdir(data_path) if isfile(join(data_path, f))]

# load file name test dataset
file_names_test = [splitext(f)[0] for f in listdir(data_path_test) if isfile(join(data_path_test, f))]


# For training
print(file_names[0])

# For test
print(file_names_test[0])


# Split the image naming in wirter, page and fragment
# For training
file_names_parts = [i.split("_") for i in file_names]
# For test
file_names_test_parts = [i.split("_") for i in file_names_test]


file_names_parts[0]


# For training
df = pd.DataFrame.from_records(file_names_parts,columns=['writer_id', 'page_id','fragment_id'])
print('Within the train dataset there are {} unique writers'.format(df.nunique()[0])) 


# For test
df_test = pd.DataFrame.from_records(file_names_test_parts,columns=['writer_id', 'page_id','fragment_id'])
print('Within the test dataset there are {} unique writers'.format(df_test.nunique()[0]))


pages_per_writer = df.groupby('writer_id')['page_id'].nunique()
pages_per_writer


# Get more information about the distribution of pages within the dataset
mean = pages_per_writer.mean()
median = pages_per_writer.median()
mu = pages_per_writer.std()
max_pages = pages_per_writer.max()
min_pages = pages_per_writer.min()

print('Every writer has {} pages on avarrage while the median is {} and the standard deviation is {}. The writer with the most pages has written {}. The writer with the least pages has written just {}page'.format(mean,median,mu,max_pages,min_pages))


pages_per_writer_value_counts = pages_per_writer.value_counts().to_frame().sort_values("page_id")
pages_per_writer_value_counts['index'] = pages_per_writer_value_counts.index
print(pages_per_writer_value_counts)


fig = plt.figure(figsize=set_size(width))
ax = sns.barplot(x ='index',y='page_id',data=pages_per_writer_value_counts, palette='Blues_r',order = pages_per_writer_value_counts.index)
ax.set_xlabel('Writers')
ax.set_ylabel('Pages')
ax.set_title('How many pages do the writers have?') 
fig.savefig('How many pages do the writers have.svg', format='svg', bbox_inches='tight',cmap='gray')


fragments_per_page = df.groupby('page_id')['fragment_id'].nunique()
fragments_per_page


# Get more information about the distribution of fragments within the dataset
mean = fragments_per_page.mean()
median = fragments_per_page.median()
mu = fragments_per_page.std()
max_pages = fragments_per_page.max()
min_pages = fragments_per_page.min()

print('Every page has {} fragments on avarrage while the median is {} and the standard deviation is {}. The page with the most fragments has {} fragments. The page with the least fragments has just {} fragment'.format(mean,median,mu,max_pages,min_pages))


fragments_per_page_value_counts = fragments_per_page.value_counts().to_frame()
fragments_per_page_value_counts['fragments'] = fragments_per_page_value_counts.index
fragments_per_page_value_counts = fragments_per_page_value_counts.sort_values('fragments')
print(fragments_per_page_value_counts)


fig = plt.figure(figsize=set_size(width))
ax = sns.barplot(x ='fragments',y='fragment_id',data=fragments_per_page_value_counts, palette='Blues_r',order = fragments_per_page_value_counts.index)
ax.set_xlabel('Fragments per Page')
ax.set_ylabel('Pages')
ax.set_title('How many pages do the writers have?') 
fig.savefig('How often do pages occur with x fragments?.svg', format='svg', bbox_inches='tight',cmap='gray')


sns.set(style="white")
fig = plt.figure(figsize=(21, 21))
fig.tight_layout(pad=0.4, w_pad=5, h_pad=2)
columns = 3
rows = 3

# ax enables access to manipulate each of subplots
ax = []

for i in range(columns*rows):
    # define filename
    filename = data_path +'/' + file_names[i] + '.jpg'
    img = Image.open(filename)
    # create subplot and append to ax
    ax.append( fig.add_subplot(rows, columns, i+1) )
    # set title
    ax[-1].set_title('wrtiter: '+file_names_parts[i][0] +'\n' + 'page: '+file_names_parts[i][1] + '\n' + 
    'fragment: '+file_names_parts[i][2])  
    #plot the img
    plt.imshow(img)

plt.show()  # finally, render the plot


# get a random writer which has three pages
writers_w_three_pages = pages_per_writer[pages_per_writer ==3]
rand_writer = writers_w_three_pages.to_frame().sample().index[0]
print(rand_writer)


# get df with writers work items
writer_df = df[df['writer_id'] == rand_writer]
writer_df = writer_df.astype(int)
writer_df = writer_df.sort_values(['page_id','fragment_id'])


# drop writer coll because it it not neccesary
writer_df = writer_df.drop(['writer_id'], axis=1)


# drop writer and add a col for coll id (row id = fragment_id)
_,idx = np.unique(writer_df['page_id'],return_inverse=True) 
writer_df['col_id'] = idx
writer_df['row_id'] = df['fragment_id']
writer_df = writer_df.set_index(['col_id','row_id'])



# get value of nrows by the number of fragments
nrows = writer_df['fragment_id'].max() +1
print(nrows)


ncols = 3  # array of sub-plots
figsize = [21, nrows * 7]     # figure size, inches


# create figure (fig), and array of axes (ax)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
fig.tight_layout(pad=0.4, w_pad=5, h_pad=2)

for i, axi in enumerate(ax.flat):
    # get indices of row/column
    rowid = i // ncols
    colid = i % ncols
    current_row = writer_df.query("col_id == {} and row_id == '{}'".format(colid,rowid))
    try:
        current_page = current_row['page_id'].astype(str).values[0]
        current_fragment = current_row['fragment_id'].astype(str).values[0]
        filename = data_path +'/' + rand_writer + '_' + current_page + '_' + current_fragment + '.jpg'
        img = Image.open(filename)    
        axi.imshow(img)
        # write row/col indices as axes' title for identification
        axi.set_title('Writer: {}\nPage: {}\n Fragment: {}'.format(rand_writer,current_page,current_fragment))
    except:
        axi.set_visible(False)
        
plt.show()

fig.savefig('readme-data.svg', format='svg', bbox_inches='tight',cmap='gray')



