import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pickle

df = pd.read_csv("data.csv", sep=";")


#df = df.drop('0', axis=1)

df = df.sample(frac=1)

X = df.drop('0.412', axis=1)
y = df['0.412']
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(X_train.shape)


#print(X_train.shape)
#clf = GaussianNB()
clf = MLPClassifier(solver='adam', activation="relu",alpha=1e-5,hidden_layer_sizes=(1000,), random_state=1, verbose=True)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('accuracy',accuracy_score(y_test, y_pred))

# save the model to disk
filename = 'model.sav'
pickle.dump(clf, open(filename, 'wb'))
