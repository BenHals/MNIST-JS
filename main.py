from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import backend as K
from keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy

dataset = datasets.load_iris()
X = dataset.data
Y = dataset.target

Y = to_categorical(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


model = Sequential()
model.add(Dense(64, input_dim=4, activation ='relu'))
model.add(Dense(18, activation ='relu'))
model.add(Dense(9, activation ='relu'))
model.add(Dense(3, activation ='softmax'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=150, validation_data=(x_test, y_test))

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('model.h5')

print(model.predict(numpy.array([[4, 3, 1, 0.2],])))