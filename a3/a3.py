from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())



def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    movies['tokens'] = movies['genres'].apply(lambda x: tokenize_string(x))
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    
    vocab = defaultdict(lambda:len(vocab))
    tokens_list = []
    #Created the vocabulary
    for tokens in movies['tokens']:
        tokens_list += tokens
    for tokens in sorted(tokens_list):
        vocab[tokens]
    N = len(movies)
    df = defaultdict(int)
    #We need to create a dictionary for each term with frequency
    for term in vocab.keys():
        for tokens in movies['tokens']:
            if term in tokens:
                df[term] += 1
    #freq of term i in document d
    tf = defaultdict(lambda:Counter())
    for i, tokens in enumerate(movies['tokens']):
        tf[i].update(tokens)
    
    features = []
    for i, tokens in enumerate(movies['tokens']):
        feature_array = np.zeros(shape = (1, len(vocab)))
        for token in tokens:
            #print(token)
            feature_array[0, vocab[token]] = tf[i][token] / tf[i].most_common(1)[0][1] * math.log10(N / df[token])
        features.append(csr_matrix(feature_array))
    movies['features'] = features
    return movies , vocab
        
    
        


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]

def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    cosine_val = np.dot(a,b.T) / (np.sqrt(a.multiply(a).sum(1)) * np.sqrt(b.multiply(b).sum(1)))
    return float(cosine_val[0 , 0])

def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    movie_similarity = defaultdict(list)
    pred_ratings = []
    for index_test, test_row in ratings_test.iterrows():
        weighted_avg = 0
        corr_sum = 0
        ratings_sum = 0
        feature_movie_to_rate = movies.loc[movies.movieId == test_row.movieId, 'features'].iloc[0]
        for index_train, train_row in ratings_train[ratings_train.userId==test_row.userId].iterrows():
            
            ratings_sum += train_row.rating
            train_movie = movies.loc[movies.movieId == train_row.movieId, 'features'].iloc[0]
            if cosine_sim (feature_movie_to_rate, train_movie) > 0:
                corr_sum += cosine_sim (feature_movie_to_rate, train_movie)
                weighted_avg += (cosine_sim (feature_movie_to_rate, train_movie) * train_row.rating)
            
                #print(cosine_sim (feature_movie_to_rate, train_movie))
                #print(train_row.rating)
                #print(weighted_avg)
        if corr_sum != 0:
            weighted_avg = round((weighted_avg / corr_sum), 1)
        else:
            weighted_avg = round((ratings_sum / len(ratings_train[ratings_train.userId==test_row.userId])), 1)
 
        pred_ratings = np.append(pred_ratings, weighted_avg)
    return pred_ratings

def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
