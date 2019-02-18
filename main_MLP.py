import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from Classes.Dataset import Dataset
from Preprocess import MovieReviewDatasetPreprocessing as movie
from Preprocess import AmazonPreprocessing as amazon
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
import pandas as pd

# parameters
from Preprocess.preProcessing import to_process

percent_of_train_test = 0.3
num_of_features = 3000
features_mode = 'intersec' # choose 'intersec' or 'union' to select if we are going to use the both features together or a insersec of both

# obtendo todos os dados de filmes e de produtos
with open('movie_train', 'rb') as fp:
    data_movies_treino = pickle.load(fp)
with open('movie_test', 'rb') as fp:
    data_movies_teste = pickle.load(fp)
with open('amazon_dataset', 'rb') as fp:
    data_amazon_original = pickle.load(fp)

x_train_original = data_movies_treino.docs
y_train_original = data_movies_treino.labels
x_test_original = data_movies_teste.docs
y_test_original = data_movies_teste.labels


# obtendo o chi2 dos dois datasets afim de
# se obter uma interseção de features
for pos in ('nouns', 'adjectives', 'nouns+adjectives', 'nouns+adjectives+adverbs', 'partial', 'all'):
    results = []

    data_amazon = Dataset()
    data_amazon.labels = data_amazon_original.labels

    data_amazon.docs = to_process(data_amazon_original.docs, pos)
    x_train = to_process(x_train_original, pos)
    x_test = to_process(x_test_original, pos)

    for features_mode in ('intersec', 'union'):
        for num_of_features in (1000, 2000, 3000, 5000, 7000, 8000, 10000, 15000, 20000):
            #print("num of features: ", num_of_features, "\nmode: ", features_mode)
            # CHI AMAZON
            cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
            x_amazon = cv.fit_transform(data_amazon.to_string(data_amazon.docs))

            chi_stats, p_vals = chi2(x_amazon, data_amazon.labels)
            chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats)),
                             key=lambda x: x[1], reverse=True)[0:num_of_features]

            features_amazon = []
            for chi in chi_res:
                features_amazon.append(chi[0])

            # CHI MOVIES
            cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
            x_movie = cv.fit_transform(data_amazon.to_string(x_train))

            chi_stats, p_vals = chi2(x_movie, y_train_original)
            chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats
                                   )), key=lambda x: x[1], reverse=True)[0:num_of_features]


            features_movie = []
            for chi in chi_res:
                features_movie.append(chi[0])


            # verificando se as features serão uma união dos dois datasets ou uma intersecção

            features = []

            if features_mode == 'intersec':
                for feature in features_amazon:
                    if feature in features_movie:
                        features.append(feature)
            else:
                for feature in features_amazon:
                    if feature not in features_movie:
                        features_movie.append(feature)
                features = features_movie


            #print("Top " + str(features.__len__()) + " features according to chi square test:")

            #print(features)

            # Fazendo o tf-idf já com as features
            cv = TfidfVectorizer(smooth_idf=True, min_df=3, norm='l1', vocabulary=features)
            x_train_tfidf = cv.fit_transform(data_amazon.to_string(x_train)) # tfidf de treino, y_train é o vetor de label
            x_test_tfidf = cv.fit_transform(data_amazon.to_string(x_test)) # tfidf de teste, y_test é o vetor de labels

            # x_train = pd.DataFrame(x_train)
            # x_test = pd.DataFrame(x_test)

            mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                                beta_1=0.9, beta_2=0.999, early_stopping=False,
                                epsilon=1e-08, hidden_layer_sizes=(5, 2),
                                learning_rate='constant', learning_rate_init=0.001,
                                max_iter=200, momentum=0.9, n_iter_no_change=10,
                                nesterovs_momentum=True, power_t=0.5, random_state=1,
                                shuffle=True, solver='lbfgs', tol=0.0001,
                                validation_fraction=0.1, verbose=False, warm_start=False)
            mlp.fit(x_train_tfidf, y_train_original)
            mlp_predic = mlp.predict(x_test_tfidf)
            mlp_accu = accuracy_score(y_test_original, mlp_predic)

            results.append([mlp_accu, features_mode, len(features)])

        results.sort(key=lambda x: x[0], reverse=True)

    print(results[0], ' ', results[1])
