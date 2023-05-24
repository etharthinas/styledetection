import pandas as pd
blog=pd.read_csv('blogtext.csv')

from sentence_transformers import SentenceTransformer
model=SentenceTransformer('all-mpnet-base-v2')
embedding=model.encode(blog['text'])

from tqdm import tqdm
n_gram_range = (3, 3)
top_n = 5
style_embedding=[]
model=SentenceTransformer('all-mpnet-base-v2')
for i in tqdm(range(len(blog))):
    count = CountVectorizer(ngram_range=n_gram_range,stop_words=None).fit([blog['text'][i]])
    candidates = count.get_feature_names_out()
    candidate_embeddings = model.encode(candidates)
    distances=cosine_similarity([embedding[i]], candidate_embeddings)
    keywords_embeddings = np.array([candidate_embeddings[index] for index in distances.argsort()[0][-top_n:]])
    mean_keywords=np.mean(keywords_embeddings,axis=0)
    style_embedding.append(embedding[i]-mean_keywords)

from sklearn.cluster import KMeans
cluster=4
kmeans=KMeans(n_clusters=cluster)
style=np.array(style_embedding)
kmeans.fit(style)
label=kmeans.labels_
