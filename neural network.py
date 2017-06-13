from keras.datasets import mnist
import numpy
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.utils import np_utils


(X_train,y_train),(X_test,y_test) = mnist.load_data()

# fix random seed for reproducibility of our results
seed = 7
numpy.random.seed(seed)


num_pixel = X_train.shape[1]*(X_train.shape[2])
#let's see our dataset's shape
print(X_train.shape)


X_train = X_train.reshape(X_train.shape[0],num_pixel).astype('float32')
X_test = X_test.reshape(X_test.shape[0],num_pixel).astype('float32')


X_train = X_train/255
X_test = X_test/255


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


#we are done with the preprocessing. Now we define our model structure
def nnmodel():
    model = Sequential()
    model.add(Dense(num_pixel,input_dim=num_pixel, init='normal', activation='relu'))
    model.add(Dense(num_classes,init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#instanciate the model
model = nnmodel()

#train the model
model.fit(X_train,y_train,batch_size=200,nb_epoch=10)

#test the accuracy of our model
score = model.evaluate(X_test,y_test,batch_size=200,verbose=0)
print(score[1]*100)