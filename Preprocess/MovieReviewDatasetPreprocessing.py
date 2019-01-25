import pickle
from Classes.Data import Data
from Classes.Dataset import Dataset
from Preprocess.preProcessing import to_process


def genRottenData():
    dataset = Dataset()

    src = ['.neg', '.pos']
    i = 0
    j = 0

    for a in src:
        arq = open('Preprocess/movie/rt-polarity' + a, 'r')

        while True:
            try:
                # Reading labels and data
                text = arq.readline()
                label = i

                if text == "":
                    break

                data = Data(doc=text, label=label)
                dataset.add(data)
            except EOFError:
                break
        i = i + 1

    dataset.docs = to_process(dataset.docs)

    #with open('Preprocess/rotten_dataset', 'wb') as fp:
    #   pickle.dump(dataset, fp)
    return dataset


def getData():
    rottendataset = Dataset()
    try:
        with open('Preprocess/rotten_dataset', 'rb') as fp:
            rottendataset = pickle.load(fp)
    except:
        rottendataset = genRottenData()

    return rottendataset
