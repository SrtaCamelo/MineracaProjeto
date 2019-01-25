import nltk

def getDictionary(data, dict_size=-1, pos_filter='all'):
    print("gettin dictionary")
    vocabulary = {}
    for document in data:
        for word in document:
            if word in vocabulary:
                j = vocabulary[word]
                j = j + 1
                vocabulary[word] = j
            else:
                vocabulary[word] = 1

    dictionary = []
    for word in vocabulary:
        dictionary.append([word, vocabulary[word]])

    dictionary.sort(key=lambda x: x[1], reverse=True)

    if dict_size > 0:
        dictionary = dictionary[:dict_size]
    else:
        dict_size = int(len(dictionary)/7)
        dictionary = dictionary[:dict_size]

    result = [word[0] for word in dictionary]

    if pos_filter == 'all':
        return result
    elif pos_filter == 'parcial':
        pos_tags = nltk.pos_tag(result)
        result = []
        for word in pos_tags:
            if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS' or word[1] == 'RB' or word[1] == 'RBR' or word[1] == 'RBS' or word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or word[1] == 'NNPS':
                result.append(word[0])
    else:
        pos_tags = nltk.pos_tag(result)
        result = []
        for word in pos_tags:
            if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS':
                result.append(word[0])

    return result


# verify if a word has a tendency to appear in a kind of label
# getting the percent of its occurence in positive and negative
# documents. if the percentage is near to 0.5, it's a neutral word.
# the limit of percentage to define is passed as a parameter.
def identifyEmotional(word_occurence, percentage):
    print("gettin emotional words")
    size = len(word_occurence)
    emotional_words = []

    for i in range(size):
        neg = word_occurence[i][0]
        pos = word_occurence[i][1]

        total = neg + pos
        neg = neg / total
        pos = pos / total

        if neg >= percentage:
            emotional_words.append([0, round(neg, 2), round(pos, 2)])
        elif pos >= percentage:
            emotional_words.append([1, round(neg, 2), round(pos, 2)])
        else:
            emotional_words.append([-1, round(neg, 2), round(pos, 2)])

    return emotional_words

# this function produces a list of lists containing a 2-dimensional vector
# for each word in dictionary. the vector V represents V[0] the percentile
# of appearance of that word in negative documents, and V[1], in positive.

def getEmotionalWords(data, label, dict_size=-1, word_precision=0.6, pos_filter='all'):
    size = len(data)

    dictionary = getDictionary(data, dict_size, pos_filter)

    word_occurence = []
    print(1)
    print(dictionary.__len__())
    j = 1
    for word in dictionary:
        wordVec = [0, 0]
        for i in range(size):
            if word in data[i]:
                wordVec[label[i]] = wordVec[label[i]] + 1
        word_occurence.append(wordVec)
        j = j+1
    emotional_words = identifyEmotional(word_occurence, word_precision)
    result = []

    for i in range(len(emotional_words)):
        if emotional_words[i][0] != -1:
            result.append(dictionary[i])
    print(result.__len__())
    return result
