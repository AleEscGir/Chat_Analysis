import time
import numpy as np
import matplotlib.pyplot as plt
from reduction import reduction
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
import matplotlib.cm as cm

import warnings

warnings.filterwarnings("ignore")

def run_algorithms():

    outliers_fraction = 0.15

    non_supervised_algorithms = [
        ("One Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel='rbf', gamma=0.1)),
        ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=42))
    ]

    supervised_algorithms = [("GaussianNB", GaussianNB()),
                            ("KNeighborsClassifier", KNeighborsClassifier()),
                            ("DecisionTreeClassifier", DecisionTreeClassifier(criterion = "entropy")),
                            ("RandomForestClassifier", RandomForestClassifier(criterion = "entropy"))]

    names, dataset, y = reduction()

    rng = np.random.RandomState(42)

    count = 0 


    # Puesta en marcha de los algoritmos de aprendizaje no supervisado:


    for X in dataset:
        for name,algorithm in non_supervised_algorithms:
            print(name + '\n')
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()

            y_predict = algorithm.fit(X).predict(X)
            print("y_predict \n" + str(y_predict))

        
            # Calcular la homogeneidad y la integridad de los clusters.
            
            homogeneity = metrics.homogeneity_score(y[count], y_predict)
            
            completeness = metrics.completeness_score(y[count], y_predict)
            
            # Calcular el coeficiente de coeficiente de Silhouette para cada muestra.
        
            s = metrics.silhouette_samples(X, y_predict)
            
            # Calcule el coeficiente de Silhouette medio de todos los puntos de datos.

            s_mean = metrics.silhouette_score(X, y_predict)
            
            # Para la configuración de los graficos -----------------------------------------------------------------------------------
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
            
            # Configura el gráfico.
            plt.suptitle('Silhouette analysis ' + name + ' : {}'.format(2),
                        fontsize=14, fontweight='bold')
            
            # Configura el 1er subgrafico.
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
            ax1.set_xlim([-3, 3])
            ax1.set_ylim([0, len(X) + (7) * 10])
            
            # Configura el 2do subgrafico.
            plt.suptitle('Silhouette analysis ' + name + ' : ' + '\n Homogeneity: {}, Completeness: {}, Mean Silhouette score: {}'.format(homogeneity,
                                                                                                completeness,
                                                                                                s_mean))
            
            
            # Para el 1er subgráfico ------------------------------------------------------------------------------------------
            
            # Grafica el coeficiente de Silhouette para cada muestra.
            cmap = cm.get_cmap("Spectral")
            y_lower = 10
            for i in range(2):
                ith_s = s[y_predict == i]
                ith_s.sort()
                size_i = ith_s.shape[0]
                y_upper = y_lower + size_i
                color = cmap(float(i) / 2)
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_s,
                                facecolor=color, edgecolor=color, alpha=0.7)
                ax1.text(-0.05, y_lower + 0.5 * size_i, str(i))
                y_lower = y_upper + 10
                
            # Trazar el coeficiente de silueta medio utilizando la línea discontinua vertical roja.
            ax1.axvline(x=s_mean, color="red", linestyle="--")
            
            # Para el 2do subgráfico 
            
            # Grafica las predicciones
            colors = cmap(y_predict.astype(float) / 2)
            ax2.scatter(X[:,0], X[:,1], c=colors)
            
            plt.show()

        count += 1


    ##################################################################################################################

    # Puesta en marcha de los algoritmos de aprendizaje supervisado

    y_for_evaluated = []

    for i in range(len(dataset)):
        y_for_evaluated.append([i] * len(dataset[i]))

    X_for_supervised = []
    y_for_supervised = []

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            X_for_supervised.append(dataset[i][j])
        y_for_supervised = y_for_supervised + y_for_evaluated[i]


    X_train, X_test, y_train, y_test = train_test_split(X_for_supervised, y_for_supervised, train_size = 0.7)


    for name, algorithm in supervised_algorithms:
        
        algorithm.fit(X_train, y_train)
        
        if len(names) < 6:
            disp = plot_confusion_matrix(algorithm, X_test, y_test,
                                    display_labels=names,
                                    cmap=plt.cm.Blues,
                                    normalize='true')
            disp.ax_.set_title("Algorithm Used: " + name + "\n" + "General_Score: " + str(algorithm.score(X_test, y_test)))
            
            plt.show()
        else:
            print(name + " has score: " + str(algorithm.score(X_test, y_test)))

