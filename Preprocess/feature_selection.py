from scipy.stats import chi2
from sklearn.feature_extraction.text import CountVectorizer

from Classes.Dataset import Dataset


def to_string(data):
    new_docs = []

    for doc in data:
        text = ""
        for word in doc:
            text = text + " " + word
        new_docs.append(text)

    return new_docs

def get_from_chi2(data, labels, num_of_features):
    data = to_string(data)
    print(data)

    cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
    x_train = cv.fit_transform(data)
    print(x_train.shape)

    # obtendo chi2
    chi_stats, p_vals = chi2(x_train, labels)

    chi_res = sorted(list(zip(cv.get_feature_names(), chi_stats
                              )), key=lambda x: x[1], reverse=True)[0:num_of_features]

    print("Top " + str(num_of_features) + " features according to chi square test:")
    print(chi_res)
    print(len(chi_res))

    features = []
    for chi in chi_res:
        features.append(chi[0])

    return features
