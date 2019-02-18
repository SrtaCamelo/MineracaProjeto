import string

import gensim
import nltk
from nltk import WordNetLemmatizer
from stemming.porter2 import stem
from keras.preprocessing.text import text_to_word_sequence

lemmatizer = WordNetLemmatizer()
punctuation = "[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]:"

def to_process(docs, pos):
    
    # Reading stop-words
    arq = open('Preprocess/sw.txt', 'r')
    stopWords = arq.read()
    stopWords = nltk.word_tokenize(stopWords)

    new_docs = []

    for text in docs:

        # Tokenizing, stemming and lemmatizing the documents
        text = gensim.parsing.stem_text(text)
        tokens = nltk.word_tokenize(text)
        stems = [stem(word) for word in tokens]
        lemma = [lemmatizer.lemmatize(word) for word in stems]

        result = []

        # Removing stop words
        for word in lemma:
            if word not in stopWords and word not in punctuation:
                result.append(word)
        
        # POS filter: only adverbs, adjectives and nouns
        pos_tags = nltk.pos_tag(result)
        result_pos = []

        if pos == 'nouns':
            for word in pos_tags:
                if word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or \
                        word[1] == 'NNPS':
                    result_pos.append(word[0])

        elif pos == 'adjectives':
            for word in pos_tags:
                if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS':
                    result_pos.append(word[0])

        elif pos == 'nouns+adjectives':
            for word in pos_tags:
                if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS' or word[1] == 'RB' or \
                        word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or \
                        word[1] == 'NNPS':
                    result_pos.append(word[0])

        elif pos == 'nouns+adjectives+adverbs':
            for word in pos_tags:
                if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS' or word[1] == 'RB' or \
                        word[1] == 'RBR' or word[1] == 'RBS' or word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or \
                        word[1] == 'NNPS':
                    result_pos.append(word[0])

        elif pos == 'partial':
            for word in pos_tags:
                if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS' or word[1] == 'RB' or \
                        word[1] == 'RBR' or word[1] == 'RBS' or word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or \
                        word[1] == 'NNPS' or word[1] == 'VB' or word[1] == 'VBD' or word[1] == 'VBG' or \
                        word[1] == 'VBN' or word[1] == 'VBP' or word[1] == 'VBZ':
                    result_pos.append(word[0])
        else:
            result_pos = result
        new_docs.append(result_pos)

    return new_docs
