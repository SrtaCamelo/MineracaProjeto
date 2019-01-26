from sklearn.feature_extraction.text import CountVectorizer
from Preprocess import MovieReviewDatasetPreprocessing as movie
from Preprocess import AmazonPreprocessing as amazon
from sklearn.feature_selection import chi2

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

print(features)
print(features.__len__())

# Fazendo o tf-idf já com as features

cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, vocabulary=features)
x_train = cv.fit_transform(data_movies.to_string(x_train)) # tfidf de treino, y_train é o vetor de label
x_test = cv.fit(data_movies.to_string(x_test)) # tfidf de teste, y_test é o vetor de labels

