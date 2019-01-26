
#Machine Learning L02
#k-fold evaluation
#SrtaCamelo

#----------------Imports--------------------
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from scipy import stats

#--------------Structures for Score Validation----------
tree_acuracy = []
nb_acuracy = []
knn_1_acuracy = []
wknn_1_acuracy = []
knn_3_acuracy = []
wknn_3_acuracy = []
knn_5_acuracy = []
wknn_5_acuracy = []
svm_linear_acuracy = []
svm_rbf_acuracy = []
logi_acuracy = []
neural_acuracy = []

#------------- Define Functions----------
#---------------Gosset's hipotesis calculatiom---
def tstudant_hipotesis(a,b):
    t2, p2 = stats.ttest_ind(a, b)
    return t2,p2
#--------------Log Generator (Could have been saved in a txt file, but naaaah)--------------
def showHipotesis():
    print("KNN-1 e KNN-3")
    t2,p2 = tstudant_hipotesis(knn_1_acuracy,knn_3_acuracy)
    print("T-value:",t2,"P-value:",p2)
    print("KNN-1 e KNN-5")
    t2, p2 = tstudant_hipotesis(knn_1_acuracy, knn_5_acuracy)
    print("T-value:", t2, "P-value:", p2)
    print("KNN-3 e KNN-5")
    t2, p2 = tstudant_hipotesis(knn_3_acuracy, knn_5_acuracy)
    print("T-value:", t2, "P-value:", p2)

    print("WKNN-1 e WKNN-3")
    t2, p2 = tstudant_hipotesis(wknn_1_acuracy, wknn_3_acuracy)
    print("T-value:", t2, "P-value:", p2)
    print("WKNN-1 e WKNN-5")
    t2, p2 = tstudant_hipotesis(wknn_1_acuracy, wknn_5_acuracy)
    print("T-value:", t2, "P-value:", p2)
    print("WKNN-3 e WKNN-5")
    t2, p2 = tstudant_hipotesis(wknn_3_acuracy, wknn_5_acuracy)
    print("T-value:", t2, "P-value:", p2)

    print("SVM-LINEAR e SVM-RBF")
    t2,p2 = tstudant_hipotesis(svm_linear_acuracy,svm_rbf_acuracy)
    print("T-value:", t2, "P-value:", p2)
def showHipotesis2():
    print("KNN-1 e NAIVE BAYES")
    t2,p2 = tstudant_hipotesis(knn_1_acuracy,nb_acuracy)
    print("T-value:",t2,"P-value:",p2)
    print("KNN-1 e Reg - log")
    t2, p2 = tstudant_hipotesis(knn_1_acuracy, logi_acuracy)
    print("T-value:", t2, "P-value:", p2)
    print("Naive Bayes e Reg - logi")
    t2, p2 = tstudant_hipotesis(nb_acuracy, logi_acuracy)
    print("T-value:", t2, "P-value:", p2)
#----------Score Acuracy-----------------
"""
Parametros: lista contendo as 50 acuracias de teste
Retorno: Média, Desvio padrão e mediana das acurácias
"""
def scores(acuracy_list):
    #scaler = StandardScaler()
    series = pd.Series(acuracy_list)
    media = series.mean()
    dp = series.std()
    mediana = series.median()
    return media,dp,mediana

#---------Função Fuleca só pra gerar o Log-----
def showStatistics(acuracy_list):
    mean, dp, median = scores(acuracy_list[0])
    print("###-------Algoritm:-----###",acuracy_list[1],"\n")
    print("#---Mean: ", mean,"\n")
    print("#---DP:", dp,"\n")
    print("#---Median:", median,"\n")

#--------------Função extramente (e desnecessariamente) redundante que chama todos os classificadores-------
"""
Parametros: x_train (conjunto de features de cada case para treino)
            y_train (classificação de cada case do treino)
            x_test (conjunto de features de cada case para teste)
            y_test (classificação de cada case do teste)
"""
#------------- Run all Algoritms for given dataSet------------
def call_classifiers(x_train, y_train, x_test, y_test):
#---------------Training Models-------------
    #----------------Decision Tree----------

    tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
    max_depth=3, min_samples_leaf=5)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    tree_accu = accuracy_score(y_test, y_pred)
    tree_acuracy.append(tree_accu)

    #---------------Naive Bayes-------------

    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    gnb_predic = gnb.predict(x_test)
    gnb_accu = accuracy_score(y_test,gnb_predic)
    nb_acuracy.append(gnb_accu)

    #--------------KNN - All variations-----

    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(x_train,y_train)
    knn1_predic = knn1.predict(x_test)
    knn1_accu = accuracy_score(y_test,knn1_predic)
    knn_1_acuracy.append(knn1_accu)
    
    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3.fit(x_train,y_train)
    knn3_predic = knn3.predict(x_test)
    knn3_accu = accuracy_score(y_test,knn3_predic)
    knn_3_acuracy.append(knn3_accu)

    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn5.fit(x_train,y_train)
    knn5_predic = knn5.predict(x_test)
    knn5_accu = accuracy_score(y_test,knn5_predic)
    knn_5_acuracy.append(knn5_accu)

    #--------------WKNN - All variations-----
    wknn1 = KNeighborsClassifier(n_neighbors=1,weights='distance')
    wknn1.fit(x_train, y_train)
    wknn1_predic = wknn1.predict(x_test)
    wknn1_accu = accuracy_score(y_test, wknn1_predic)
    wknn_1_acuracy.append(wknn1_accu)

    wknn3 = KNeighborsClassifier(n_neighbors=3,weights='distance')
    wknn3.fit(x_train, y_train)
    wknn3_predic = wknn3.predict(x_test)
    wknn3_accu = accuracy_score(y_test, wknn3_predic)
    wknn_3_acuracy.append(wknn3_accu)

    wknn5 = KNeighborsClassifier(n_neighbors=5,weights='distance')
    wknn5.fit(x_train, y_train)
    wknn5_predic = wknn5.predict(x_test)
    wknn5_accu = accuracy_score(y_test, wknn5_predic)
    wknn_5_acuracy.append(wknn5_accu)

    #----------------SVM Linear--------------

    svm_linear = svm.SVC(C=1, kernel='linear')
    svm_linear.fit(x_train, y_train)
    svm_predic = svm_linear.predict(x_test)
    svml_accu = accuracy_score(y_test,svm_predic)
    svm_linear_acuracy.append(svml_accu)


    # ----------------SVM RBF--------------

    svm_rbf = svm.SVC(C=1, kernel='rbf')
    svm_rbf.fit(x_train, y_train)
    svm_predic = svm_rbf.predict(x_test)

    svmr_accu = accuracy_score(y_test, svm_predic)
    svm_rbf_acuracy.append(svmr_accu)
    
    #Logistic Regression
    logi = LogisticRegression(random_state=0, solver='lbfgs',
                        multi_class='multinomial')
    logi.fit(x_train,y_train)
    logi_predic = logi.predict(x_test)

    logi_accu = accuracy_score(y_test,logi_predic)
    logi_acuracy.append(logi_accu)

    #MLP

    mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
    mlp.fit(x_train, y_train)
    mlp_predic = mlp.predict(x_test)

    mlp_accu = accuracy_score(y_test, mlp_predic)
    neural_acuracy.append(mlp_accu)
#--------------Shuffle data an Split Features-------------
def prepareDataSet(datapath):
    #Shuffle
    data = pd.read_csv(datapath) 
    data = data.sample(frac=1)
    return data
"""
Parametros: data : Pandas DataFrame
Essa função divide o dataSet randomizado em 10 folds,
repassando o conjunto de treinamento e teste pros classificadores.

"""
#--------------K-fold method (k = 10)---------------------
def kfold_call(data,n_start, n_final,key):

    #---------Split Class column from features-------------

    X = data.iloc[:,n_start:n_final]
    Y = data[key]

    kf = KFold(n_splits=10, shuffle=False)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        call_classifiers(x_train, y_train, x_test, y_test)

#---------------Variables and Constants ----------------
path = "teste100.csv"

#---------------------Main Code ------------------------
data = prepareDataSet(path)
for i in range(5):
    kfold_call(data,2,len(data.columns)-1,"class")


#--------------These code lines above Do de SD, median and mean calculus---------------------------

List_ofLists =[(tree_acuracy,"Tree"),(nb_acuracy,"Naive bayes"),(knn_1_acuracy,"KNN -1"),(knn_3_acuracy,"KNN -3"),
               (knn_5_acuracy, "KNN -5"),(wknn_1_acuracy,"WKNN-1"),(wknn_3_acuracy,"WKNN-3"),(wknn_5_acuracy,"WKNN-5"),
               (svm_linear_acuracy,"SVM-Linear"),(svm_rbf_acuracy,"SVM-RBF"),(logi_acuracy,"Logistic Regression"),(neural_acuracy,"MLP")]

for acuracy_list in List_ofLists:
    #print(acuracy_list[0])
    showStatistics(acuracy_list)

#-------------This call above calls t-studant hipotesis for classifiers variations-----------------
#showHipotesis2()
