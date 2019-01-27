from sklearn.feature_extraction.text import CountVectorizer
from Preprocess import MovieReviewDatasetPreprocessing as movie
from Preprocess import AmazonPreprocessing as amazon
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# parameters
percent_of_train_test = 0.3
num_of_features = 15000
features_mode = 'intersec' # choose 'intersec' or 'union' to select if we are going to use the both features together or a insersec of both

# obtendo todos os dados de filmes e de produtos
data_movies = movie.getData()
data_amazon = amazon.getData()

# dividindo em treino e teste a base de filmes que será usada
# para a avaliação
x_train, y_train, x_test, y_test = data_movies.get_train_test(percent=percent_of_train_test, num_docs=10000)

# obtendo o chi2 dos dois datasets afim de
# se obter uma interseção de features
cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
x_amazon = cv.fit_transform(data_amazon.to_string(data_amazon.docs))

chi_stats, p_vals = chi2(x_amazon, data_amazon.labels)
chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats
                       )), key=lambda x: x[1], reverse=True)[0:num_of_features]

print("Top " + str(num_of_features) + " features according to chi square test:")

features_amazon = []
for chi in chi_res:
    features_amazon.append(chi[0])

cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
x_movie = cv.fit_transform(data_movies.to_string(x_train))

chi_stats, p_vals = chi2(x_movie, y_train)
chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats
                       )), key=lambda x: x[1], reverse=True)[0:num_of_features]

print("Top " + str(num_of_features) + " features according to chi square test:")

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
    feature = features_movie + features_amazon

#print(features)
#sprint(features.__len__())

# Fazendo o tf-idf já com as features

cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, vocabulary=features)
x_train = cv.fit_transform(data_movies.to_string(x_train)) # tfidf de treino, y_train é o vetor de label
x_test = cv.fit(data_movies.to_string(x_test)) # tfidf de teste, y_test é o vetor de labels

x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                    beta_1=0.9, beta_2=0.999, early_stopping=False,
                    epsilon=1e-08, hidden_layer_sizes=(5, 2),
                    learning_rate='constant', learning_rate_init=0.001,
                    max_iter=200, momentum=0.9, n_iter_no_change=10,
                    nesterovs_momentum=True, power_t=0.5, random_state=1,
                    shuffle=True, solver='lbfgs', tol=0.0001,
                    validation_fraction=0.1, verbose=False, warm_start=False)
mlp.fit(x_train, y_train)
mlp_predic = mlp.predict(x_test)
mlp_accu = accuracy_score(y_test, mlp_predic)
print(mlp_accu)

