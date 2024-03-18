from nn import nn
import numpy as np
import sklearn
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt

#Get digits dataset
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)
X = optical_recognition_of_handwritten_digits.data.features 
y = optical_recognition_of_handwritten_digits.data.targets
y_onehot=np.zeros((y.shape[0],10))
y_onehot[np.arange(y.shape[0]),y["class"]]=1

# Sample 80/20 train/test split
train_idx=np.random.randint(0,len(X)-1,int(len(X)*0.8))
train_X=np.array(X.iloc[list(train_idx),:])
train_y=np.array(y_onehot[train_idx])
test_idx=set(range(len(X)))-set(train_idx)
test_X=np.array(X.iloc[list(test_idx),:])
test_y=np.array(y_onehot[list(test_idx)])

#Initialize NN
NNet=nn.NeuralNetwork(
    nn_arch=[{'input_dim':64,'output_dim':16,'activation':"sigmoid"},{'input_dim':16,'output_dim':64,'activation':"relu"},{'input_dim':64,'output_dim':10,"activation":"sigmoid"}],
    lr=0.001,
    seed=37,
    batch_size=100,
    epochs=100,
    loss_function="BCE"
)

# Train NN
(train_error,val_error)=NNet.fit(train_X,train_y,test_X,test_y)
print(train_error)