from Preprocess.Emotional_Words import getDictionary
from Preprocess import IMDBPreProcessing as imdb
from Preprocess.sequencing import text_to_tfidf

# obtendo todos os dados
data = imdb.getData()

# dividindo em treino e teste
x_train, y_train, x_test, y_test = data.get_train_test(percent=0.3, num_docs=1000)


# obtendo o dicionario
dictionary = getDictionary(data=x_train)

# obtendo o tf-idf
x_train = text_to_tfidf(data=x_train,dictionary=dictionary)
x_test = text_to_tfidf(data=x_train, dictionary=dictionary)

print(x_train)
