import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import pairwise_distances
from ast import literal_eval
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

movieset = pd.read_csv("movies_metadata.csv", low_memory=False)

c = movieset['vote_average'].mean()
d = movieset['vote_count'].quantile(.60)
q_movies = movieset.copy().loc[movieset['vote_count'] >= d]



def weighted_rating(x, d=d, c=c):
    v = x['vote_count']
    r = x['vote_average']
    return (v / (v + d) * r) + (d / (d + v) * c)


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

# Define a TF-IDF Vectorizer Object. Remove all english stop words 
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
q_movies['overview'] = q_movies['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(q_movies['overview'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(q_movies.index, index=q_movies['title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[1:11]
    movie_indices = [i[0] for i in sim_score]
    return q_movies['title'].iloc[movie_indices]


credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

# q_movies = q_movies.drop([19730, 29503, 35587])
# Convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
q_movies['id'] = q_movies['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
q_movies = q_movies.merge(credits, on='id')
q_movies = q_movies.merge(keywords, on='id')

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    q_movies[feature] = q_movies[feature].apply(literal_eval)


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []


q_movies['director'] = q_movies['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']

for feature in features:
    q_movies[feature] = q_movies[feature].apply(get_list)


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i['name'].replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    q_movies[feature] = q_movies[feature].apply(clean_data)


def create_soup(x):
    return  ' '.join(x['cast']) + ' ' +5* x['director'] + ' ' + ' '.join(x['genres'])


# Create a new soup feature
q_movies['soup'] = q_movies.apply(create_soup, axis=1)

# count = CountVectorizer(stop_words='english')
# count_matrix = count.fit_transform(q_movies['soup'])
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(q_movies['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

q_movies = q_movies.reset_index()
indices = pd.Series(q_movies.index, index=q_movies['title'])


print(get_recommendations('Inception', cosine_sim2))
