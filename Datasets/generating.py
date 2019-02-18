import pickle

from Classes.Data import Data
from Classes.Dataset import Dataset
from Preprocess.MovieReviewDatasetPreprocessing import getData


def processing_amazon():
    dataset = Dataset()
    arq = open('Preprocess/AmazonDatasetTest.txt', 'r', encoding='utf-8')
    i = 0

    pos, neg = 0, 0

    while True:
        try:
            line = arq.readline()
            if line == "":
                break
            print(i)
            i = i + 1

            # Reading labels and data
            label = int(line[9])
            label = 0 if label == 1 else 1
            text = line[11:len(line) - 1]

            if label == 1:
                if pos < 5000:
                    data = Data(doc=text, label=label)
                    dataset.add(data)
                    pos = pos+1
            if label == 0:
                if neg < 5000:
                    data = Data(doc=text, label=label)
                    dataset.add(data)
                    neg = neg + 1

            if pos == 5000 and neg == 5000:
                break
        except EOFError:
            break

    with open('amazon_dataset', 'wb') as fp:
        pickle.dump(dataset, fp)

def processing_movie():
    dataset = getData()
    x_train, y_train, x_test, y_test = dataset.get_train_test(percent=0.3, num_docs=10000)

    dataset_treino = Dataset()
    dataset_treino.docs =x_train
    dataset_treino.labels = y_train

    dataset_teste = Dataset()
    dataset_teste.docs = x_test
    dataset_teste.labels = y_test

    with open('movie_train', 'wb') as fp:
        pickle.dump(dataset_treino, fp)

    with open('movie_test', 'wb') as fp:
        pickle.dump(dataset_teste, fp)

processing_movie()