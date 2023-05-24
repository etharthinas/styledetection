import pandas as pd
blog=pd.read_csv('blogtext.csv')

from sentence_transformers import SentenceTransformer
model=SentenceTransformer('all-mpnet-base-v2')
embedding=model.encode(blog['text'])

import numpy as np
from tqdm import tqdm
style_embedding=embedding
for t in tqdm(blog['topic'].unique()):
    blog_top=blog[blog['topic']==t].index.values
    c=[]
    for i in blog_top:
        c.append(embedding[i].numpy())
    mean=np.mean(c,axis=0)
    for i in blog_top:
        style_embedding[i]=embedding[i]-mean

from sklearn.cluster import KMeans
cluster=4
kmeans=KMeans(n_clusters=cluster)
kmeans.fit(style_embedding)
label=kmeans.labels_
