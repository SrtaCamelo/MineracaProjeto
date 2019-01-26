import pickle

from Preprocess.preProcessing import to_process


def getData():
    with open('Preprocess/amazon_dataset', 'rb') as fp:
        amazon = pickle.load(fp)
        amazon.docs = amazon.to_string(amazon.docs)
        amazon.docs = to_process(amazon.docs)

        return amazon
