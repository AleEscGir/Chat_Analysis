from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from vectorizer import take_data

warnings.filterwarnings("ignore")

def reduction ():

    data = take_data()
    dataset = []
    users =[]

    for user in data:
        users.append(user)
        for j in data[user]['vector_values']:
            X_features =j

            scaler = StandardScaler()
            X_features = scaler.fit_transform(X_features)

            pca = PCA(n_components=2)
            pca.fit(X_features)
            x_3d = pca.transform(X_features)
            dataset.append(x_3d)


    return users, dataset, y()

def y ():
    a = [1]*28
    al = [1]*310
    m = [1]*320
    for i in range(5,10):
        a[i] = -1
        a[len(a)-1 -i] = -1
    for i in range(3,6):
        al[i] = -1
        al[i+6] = -1
    y = [a,al,m]
    return y
