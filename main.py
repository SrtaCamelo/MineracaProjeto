from Preprocess.Emotional_Words import getDictionary
from Preprocess.MovieReviewDatasetPreprocessing import getData
from Preprocess.sequencing import text_to_tfidf, text_to_onehot

# obtendo todos os dados
from convolutional_nn import CNN_Model

data = getData()

# obtendo one hot encoding dos dados
data.docs, words_size = text_to_onehot(data.docs)

# dividindo em treino e teste
x_train, y_train, x_test, y_test = data.get_train_test(0.3, 1000)


# obtendo o dicionario
dictionary = getDictionary(data=x_train)

# obtendo o tf-idf
x_train = text_to_tfidf(x_train, dictionary)
x_test = text_to_tfidf(x_train, dictionary)

# cnn
model = CNN_Model(x_train, y_train, x_test, y_test)