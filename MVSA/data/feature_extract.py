import numpy as np
from PIL import Image
import gist
import pandas as pd
import os
import concurrent.futures
import pickle
from tqdm import tqdm
from gensim.utils import simple_preprocess

# extract labels
label_df = pd.read_csv('MVSA_Single/labelResultAll.txt', sep='\t|,',
                       engine='python', index_col=0)
label_df['text'][label_df['text'] == 'neutral'] = -1
label_df['text'][label_df['text'] == 'positive'] = 1
label_df['text'][label_df['text'] == 'negative'] = 0

to_include = (label_df['text'] != -1).values

idx_names = label_df.index.values[to_include]
labels = label_df['text'].values[to_include]

texts = []
for i in idx_names:
    txt = pd.read_csv(os.path.join('MVSA_Single/data', '{}.txt'.format(i)),
                             header=None, encoding= 'unicode_escape',
                             quoting=csv.QUOTE_NONE).values[0][0]
    # remove @'s, i.e. mentions
    txt = ' '.join([x for x in txt.split(' ') if x != 'RT'])
    txt = ' '.join([x for x in txt.split(' ') if '@' not in x])
    txt = ' '.join([x for x in txt.split(' ') if 'http' not in x])
    texts.append(txt)

for i, itm in enumerate(texts):
    texts[i] = ' '.join(simple_preprocess(itm, deacc=True))

def process_image(img_name):
    img = Image.open(os.path.join('MVSA_Single/data',
                                  '{}.jpg'.format(img_name)))
    img_array = np.array(img)

    return gist.extract(img_array).squeeze()

if __name__ == "__main__":
    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)

    with open('texts.pkl', 'wb') as f:
        pickle.dump(texts, f)

    l = labels.shape[0]
    with tqdm(total=l) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_image, arg): arg 
                        for arg in idx_names}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                arg = futures[future]
                results[arg] = future.result()
                pbar.update(1)

    with open('image_mat.pkl', 'wb') as f:
        pickle.dump(results, f)
