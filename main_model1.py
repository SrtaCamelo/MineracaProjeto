import pickle
import random
from keras import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, LSTM, Dense
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from Classes.Dataset import Dataset
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, adjusted_rand_score
from NeuralNetworks.neural_networks import hybrid
from Preprocess.preProcessing import to_process, rand_index_score

# ---------------------------------------- parameters -------------------------------------------
classifier = 'knn'
num_of_features = 5000
pos = '3'
features_mode = 'intersec'

# --------------------------------------- loading datasets ---------------------------------------
with open('Datasets/dataset_movies_10k', 'rb') as fp:
    dataset_target = pickle.load(fp)
with open('Datasets/dataset_music_10k', 'rb') as fp:
    dataset_source = pickle.load(fp)
print("Partial results\nTFIDF\nStop words, POS filter, tokenizing, lemmatizing and stemming.")
print("POS\n1: Adjectives\n2: Adverbs\n3: Nouns\n4: Verbs\n5: Adjectives and adverbs\n6: Adjectives, adverbs and nouns")

#  -------------------------------------- randomizing  -----------------------------------------
# dataset_target.docs = random.sample(dataset_target.docs, len(dataset_target.docs))
dataset_train, dataset_test, y_train, y_test = train_test_split(dataset_target.docs, dataset_target.labels,
                                                                train_size=0.8, random_state=42)
data_source = Dataset()
data_source.labels = dataset_source.labels

# ------------------------------------ preprocessing  ----------------------------------------

data_source.docs = to_process(dataset_source.docs, pos)
x_train = to_process(dataset_train, pos)
x_test = to_process(dataset_test, pos)

#  -------------------------------------- chi source  -----------------------------------------
cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
x_source = cv.fit_transform(data_source.to_string(data_source.docs))

chi_stats, p_vals = chi2(x_source, data_source.labels)
chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats)),
                 key=lambda x: x[1], reverse=True)[0:num_of_features]

features_source = []
for chi in chi_res:
    features_source.append(chi[0])

# --------------------------------------- chi target -----------------------------------------
cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
x_target = cv.fit_transform(data_source.to_string(x_train))

chi_stats, p_vals = chi2(x_target, y_train)
chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats
                          )), key=lambda x: x[1], reverse=True)[0:num_of_features]

features_target = []
for chi in chi_res:
    features_target.append(chi[0])

#  ------------------------------------- features selection  ----------------------------------

features = []

if features_mode == 'intersec':
    for feature in features_source:
        if feature in features_target:
            features.append(feature)
else:
    features = [a for a in features_target]
    for feature in features_source:
        if feature not in features:
            features.append(feature)

print(features_source)
print(features_target)
print(features)
print('Source\'s features: ', features_source.__len__())
print('Target\'s features: ', features_target.__len__())
print('Number of features after ', features_mode, ':', features.__len__())

#  ----------------------------------------- tf-idf  -----------------------------------------

cv = TfidfVectorizer(smooth_idf=True, min_df=3, norm='l1', vocabulary=features)
x_train_tfidf = cv.fit_transform(data_source.to_string(x_train))  # tfidf de treino, y_train é o vetor de label
x_test_tfidf = cv.fit_transform(data_source.to_string(x_test))  # tfidf de teste, y_test é o vetor de labels

#  -------------------------------------- classifying  ---------------------------------------

if classifier == 'cnn':
    model = Sequential()
    model.add(Embedding(500, 100, input_length=len(features)))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_tfidf, np.array(y_train), validation_split=0.4, epochs=3, verbose=1)
    scores = model.evaluate(x_test_tfidf, np.array(y_test), verbose=0)
    print("Classifier: ", classifier)
    print("POS: ", pos)
    print("Number of Features: ", len(features))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("\n-------------------------------------------------------------\n")

elif classifier == 'rnn':
    model = Sequential()
    model.add(Embedding(len(features), 100, input_length=len(features)))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_tfidf, np.array(y_train), validation_split=0.4, epochs=3, verbose=0)
    scores = model.evaluate(x_test_tfidf, np.array(y_test), verbose=0)
    print("Classifier: ", classifier)
    print("POS: ", pos)
    print("Number of Features: ", len(features))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("\n-------------------------------------------------------------\n")

elif classifier == 'mlp':
    mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(5, 2),
                        learning_rate='constant', learning_rate_init=0.001,
                        max_iter=200, momentum=0.9, n_iter_no_change=10,
                        nesterovs_momentum=True, power_t=0.5, random_state=1,
                        shuffle=True, solver='lbfgs', tol=0.0001,
                        validation_fraction=0.1, verbose=False, warm_start=False)
    mlp.fit(x_train_tfidf, y_train)
    mlp_predic = mlp.predict(x_test_tfidf)
    mlp_accu = accuracy_score(y_test, mlp_predic)
    f_measure = f1_score(y_test, mlp_predic, average='weighted')
    print(f_measure)
    print("Classifier: ", classifier)
    print("POS: ", pos)
    print("Number of Features: ", len(features))
    print(mlp_accu * 100)
    print("\n-------------------------------------------------------------\n")
    
elif classifier == 'knn':
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train_tfidf, y_train)

    predict = [neigh.predict(test) for test in x_test_tfidf]
    print(len(predict))

    f_measure = f1_score(y_test, predict, average='weighted')
    print(f_measure)
    accuracy = accuracy_score(y_test, predict)
    print(accuracy)
    confMatrix = confusion_matrix(y_test, predict)
    print(confMatrix)
