# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    """
    ###TODO
    if (keep_internal_punct):
        return re.sub('\W+\B', ' ', doc).lower().split() if doc else []
        
      
    else:
        return re.sub('\W+', ' ', doc).lower().split() if doc else []

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    for token in tokens:
        key = 'token=' + token
        feats[key] += 1
    return feats


def token_pair_features(tokens, feats, k=2):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO
    
    for window_start in range(int(len(tokens)/k)+2 ):
        
        for j in combinations(tokens[window_start:(window_start+k)], 2):
            #print(j)
            key = 'token_pair=' + j[0] + "_" + j[1] 
            feats[key] += 1
        
   

     


#neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring', 'annoying', 'failed', 'miserably'])
neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])


def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO
    feats['neg_words'] = 0
    feats['pos_words'] = 0
    for token in tokens:
        if token.lower() in neg_words:
            feats['neg_words'] += 1
        elif token.lower() in pos_words:
            feats['pos_words'] += 1
    return feats



def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    
    func_feats = defaultdict(lambda: 0)
    for function in feature_fns:
     
        function(tokens, func_feats)
    return sorted(func_feats.items())
        
       

def vectorize(tokens_list, feature_fns, min_freq=0, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
    feats = defaultdict(lambda: 0)
    check = defaultdict(lambda: 0)
    row = []
    col = []
    
    rows = []
    tokens_all = []
    #Creating vocabulary
    
    for tokens in tokens_list:
        #tokens_all = tokens_all + tokens
        rows.append(featurize(tokens, feature_fns))
    for feat in rows:
        tokens_all = tokens_all + feat
    
    #print(len(vocab))
    if (vocab is None):
        vocab = defaultdict(lambda:len(vocab))
        for feat in sorted(tokens_all):
            if feat[0] not in vocab:
                vocab[feat[0]]
    #Creating features
    #print(2)
    idx2word = {v: k for k, v in vocab.items() }
    feature_array = np.zeros(shape = (len(tokens_list), len(vocab)))
    for row_index, row_vals in enumerate(rows):
        for row_val in row_vals:
            if row_val[0] in vocab:
                col_index = vocab[row_val[0]]
                feature_array[row_index, col_index] = row_val[1]
     #checking the minimum frequency
    #print(1)
    non_zero = np.count_nonzero(feature_array, axis=0)
    index = []
    for i in range(len(non_zero)):
        if non_zero[i] < min_freq:
            index.append(i)
            del(idx2word[i])
    #for i in index:
    #print(2)  
    #print(1)
    feature_array = np.delete(feature_array,index,1)
    new_vocab = defaultdict(lambda: len(new_vocab))
    #print(3)
    for k , v in sorted(idx2word.items()):
        new_vocab[v]
    #print(2)
    
    #print(3)
    return csr_matrix(feature_array) , new_vocab


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)

def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    #print(5)
    cv = KFold(n_splits=k)
    accuracies = []
    for train_idx, test_idx in cv.split(X):
        
        clf.fit(X[train_idx], labels[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(labels[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    #print(6)
    return avg

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    token_list = []
    
    final_list = []
    clf = LogisticRegression()
    #for each punctuation
    for punc in punct_vals:
        #for each document
        tokens_list = [tokenize(d,keep_internal_punct=punc) for d in docs]
        
        #for d in docs:
         #   token_list.append(tokenize(d, keep_internal_punct=punc))
    #for every possible combination
        for i in range(1 , (len(feature_fns)+1)):
            #for each combination of feature function
            for feat_comb in combinations(feature_fns,i):
                #print(feat_comb)
                #for each value of min freq
                for freq in min_freqs:
                    accuracy_dict = {}

                    #vectorize for this combination
                    #print(feat_comb)
                    X,vocab = vectorize(tokens_list, feat_comb, freq,vocab=None)
                    #calculate accuracy
                    accuracy = cross_validation_accuracy(clf, X, labels, 5)
                    #assign dictionary values
                    accuracy_dict['features'] = feat_comb
                    #print(feat_comb)
                    accuracy_dict['punct'] = punc
                    #print(punc)
                    accuracy_dict['min_freq'] = freq
                    accuracy_dict['accuracy'] = accuracy
                    #print(accuracy)
                    final_list.append(accuracy_dict)
    return sorted(final_list, key=lambda k: k['accuracy'], reverse = True) 
    

def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    
    accuracy_values = []
    plt.figure()
    #print(results)
    for results_dict in results:
        accuracy_values.append(results_dict['accuracy'])
    
    plt.plot(sorted(accuracy_values))
    plt.savefig('accuracies.png')
    plt.show()
    

def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO
    output_list = []
    min_freq = defaultdict(list)
    punct = defaultdict(list)
    features = defaultdict(list)
    dic_list = [min_freq, punct, features]
    headings = ['min_freq', 'punct', 'features']
    
    for results_dict in results:
        for i, setting in enumerate(dic_list):
            
            setting[results_dict[headings[i]]].append(results_dict['accuracy'])
            
    for i, setting in enumerate(dic_list):
        for key, val in setting.items():
            
            tup1 = np.mean(val)
            tup2 = headings[i] +"=" + str(key)
            output_list.append((tup1, tup2))

        

    return sorted(output_list, key = lambda x:x[0], reverse = True)

def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
      
    """
    ###TODO
    feat_comb = best_result['features']
    freq = best_result['min_freq']
    punc = best_result['punct']
    clf = LogisticRegression()
    tokens_list = [tokenize(d,keep_internal_punct=punc) for d in docs]
    X,vocab = vectorize(tokens_list, feat_comb, freq)
    print(X.shape[1])
    clf.fit(X, labels)
    return clf, vocab
    

def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    # Get the learned coefficients for the Positive class.
    coef = clf.coef_[0]
    
    output_list = []
    idx2word = dict((v,k) for k,v in vocab.items())
    # Sort them in descending order.
    if label == 1:
        top_coef_ind = np.argsort(coef)[::-1][:n]
    else:
        top_coef_ind = np.argsort(coef)[:n]
    
    # Get the names of those features.
    idx2word = dict((v,k) for k,v in vocab.items())
    top_coef_terms = [idx2word[i] for i in top_coef_ind]
    top_coef = coef[top_coef_ind]
    #print(top_coef)
    for i , term in enumerate(top_coef_terms):
        output_list.append((term, top_coef[i] ))
    return output_list
    

def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    feat_comb = best_result['features']
    freq = best_result['min_freq']
    punc = best_result['punct']
    test_docs, labels = read_data(os.path.join('data', 'test'))
    tokens_list = [tokenize(d, keep_internal_punct=punc) for d in test_docs]
    
    X, vocab = vectorize(tokens_list, feature_fns =feat_comb,vocab=vocab)
    return test_docs, labels, X
    
    

def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    predictions = clf.predict(X_test)
    predicted_prob = clf.predict_proba(X_test)
    output = []
    for i, predicted_vals in enumerate(predicted_prob):
        if predictions[i] != test_labels[i]:
            if predictions[i] == 1:
                output.append((test_labels[i], predictions[i], predicted_vals[1], test_docs[i]))
            else:
                output.append((test_labels[i], predictions[i], predicted_vals[0], test_docs[i]))
    
    output = sorted(output, key=lambda tup: tup[2], reverse = True)
    for i in range(n):
        print("truth="+str(output[i][0])+" "+ "predicted=" + str(output[i][1])+ " "+"proba="+str(output[i][2]))
        print(output[i][3])
    
from sklearn.feature_extraction.text import TfidfTransformer

#Additional implementation to improve accuracy
def TF_IDF_vectorize_V2(tokens_list, feature_fns, min_freq=0, vocab=None):
    
    X, vocab = vectorize(tokens_list, feature_fns, min_freq, vocab)
    tfidf_transformer = TfidfTransformer(use_idf = True, sublinear_tf = True)
    X_train_tfidf = tfidf_transformer.fit_transform(X)
    #vocab = np.array(tfidf_transformer.get_feature_names())
    return(X_train_tfidf, vocab)

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
zipfile = ZipFile(BytesIO(url.read()))
afinn_file = zipfile.open('AFINN/AFINN-111.txt')

afinn = dict()

for line in afinn_file:
    parts = line.strip().split()
    if len(parts) == 2:
        afinn[parts[0].decode("utf-8")] = int(parts[1])

def lexicon_features_afinn(tokens, feats):
    
    feats['neg_words'] = 0
    feats['pos_words'] = 0
    for token in tokens:
        if token.lower() in afinn:
            if afinn[token.lower()] < 0 :
                feats['neg_words'] += 1
            elif afinn[token.lower()] > 0:
                feats['pos_words'] += 1
    return feats

def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
