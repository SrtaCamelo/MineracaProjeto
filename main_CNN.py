# CHI AMAZON
import pickle

import keras
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from Classes.Dataset import Dataset
from NeuralNetworks.neural_networks import separableCNN, cnn, dense
from Preprocess.preProcessing import to_process

percent_of_train_test = 0.3
num_of_features = 3000
features_mode = 'intersec' # choose 'intersec' or 'union' to select if we are going to use the both features together or a insersec of both

# obtendo todos os dados de filmes e de produtos
with open('Datasets/movie_train', 'rb') as fp:
    data_movies_treino = pickle.load(fp)
with open('Datasets/movie_test', 'rb') as fp:
    data_movies_teste = pickle.load(fp)
with open('Datasets/amazon_dataset', 'rb') as fp:
    data_amazon_original = pickle.load(fp)

x_train_original = data_movies_treino.docs
y_train_original = data_movies_treino.labels
x_test_original = data_movies_teste.docs
y_test_original = data_movies_teste.labels



# obtendo o chi2 dos dois datasets afim de
# se obter uma interseção de features
if True:
    pos1 = 'partial'
    results = []

    data_amazon = Dataset()
    data_amazon.labels = data_amazon_original.labels

    print(0)
    data_amazon.docs = to_process(data_amazon_original.docs, pos1)
    x_train = to_process(x_train_original, pos1)
    x_test = to_process(x_test_original, pos1)


    for features_mode in ('intersec'):
        for num_of_features in (2000, 3000, 5000, 7000):
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

            y_test_original = keras.utils.to_categorical(y_test_original,2)
            y_train_original = keras.utils.to_categorical(y_train_original,2)

            # x_train = pd.DataFrame(x_train)
            # x_test = pd.DataFrame(x_test)

            model = dense(len(features))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit(x_train_tfidf, y_train_original, verbose=1, epochs=3)
            score = model.evaluate(x_test_tfidf, y_test_original, verbose=1)

        print('\n \n ---------------------------------------------------------------------- \n \n')

