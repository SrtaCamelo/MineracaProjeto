import math


def text_to_binary(data, dictionary):
    print("docs to binary")
    sequences = []
    i = 0
    for doc in data:
        #print(i)
        i = i+1
        doc_sequence = []

        for word in dictionary:
            if word in doc:
                doc_sequence.append(1)
            else:
                doc_sequence.append(0)

        sequences.append(doc_sequence)

    return sequences

def text_to_tfidf(data, dictionary):
    print("docs to tfidf")
    sequences = []
    for doc in data:
        doc_sequence = []

        for word in dictionary:
            if word in doc:
                doc_sequence.append(gettf(word, doc) * getidf(word, data))
            else:
                doc_sequence.append(0)

        sequences.append(doc_sequence)

    return sequences

def gettf(word, doc):
    n_words = len(doc)
    occ = 0

    for term in doc:
        if term == word:
            occ = occ + 1

    return occ / n_words

def getidf(word, data):
    size = len(data)
    occ = 0

    for doc in data:
        if word in doc:
            occ = occ + 1

    if occ == 0:
        return 0

    return math.log2(size / occ)

def text_to_onehot(whole_data):
    print("text to one hot")
    words = {}
    i = 1

    for text in whole_data:
        for word in text:
            if word not in words:
                words[word] = i
                i = i+1

    data = []

    for text in whole_data:
        document = []
        for word in text:
            document.append(words[word])
        data.append(document)

    return data, len(words)
