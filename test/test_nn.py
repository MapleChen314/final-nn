from nn import nn, preprocess
import numpy as np
import sklearn
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike
from ucimlrepo import fetch_ucirepo 
    

def test_single_forward():
    NNet=nn.NeuralNetwork(
        nn_arch=[{'input_dim':3,'output_dim':2,'activation':"relu"},{'input_dim':2,'output_dim':3,'activation':"sigmoid"}],
        lr=0.001,
        seed=37,
        batch_size=2,
        epochs=10,
        loss_function="MSE"
    )
    X_train=np.array([[1,2,3],[4,7,3],[0,5,-1]])
    y_train=np.array([6,14,4])
    W_curr=NNet._param_dict["W1"]
    b_curr=NNet._param_dict["b1"]
    A_prev=X_train
    activation="relu"
    A_curr,Z_curr=NNet._single_forward(W_curr,b_curr,A_prev,activation)
    assert A_curr.shape==(W_curr.shape[0],A_prev.shape[1])

def test_forward():
    NNet=nn.NeuralNetwork(
        nn_arch=[{'input_dim':3,'output_dim':2,'activation':"relu"},{'input_dim':2,'output_dim':3,'activation':"sigmoid"},{'input_dim':3,'output_dim':1,'activation':"relu"}],
        lr=0.001,
        seed=37,
        batch_size=2,
        epochs=10,
        loss_function="MSE"
    )
    X_train=np.array([[1,2,3],[4,7,3],[0,5,-1]]) 
    y_train=np.array([6,14,4])
    y_hat,cache=NNet.forward(X_train)
    assert "A3" in cache.keys()
    assert y_hat.shape==(1,X_train.shape[1])
    

def test_single_backprop():
    NNet=nn.NeuralNetwork(
        nn_arch=[{'input_dim':3,'output_dim':2,'activation':"relu"},{'input_dim':2,'output_dim':3,'activation':"sigmoid"},{'input_dim':3,'output_dim':1,'activation':"relu"}],
        lr=0.001,
        seed=37,
        batch_size=2,
        epochs=10,
        loss_function="MSE"
    )
    X_train=np.array([[1,2,3],[4,7,3],[0,5,-1]])
    y_train=np.array([[6,14,4]])
    y_hat,cache=NNet.forward(X_train.T)
    grad_dict=NNet.backprop(y_train,y_hat,cache)
    print(grad_dict.keys())
    print(grad_dict["W1"].shape)
    print(grad_dict["W2"].shape)
    print(grad_dict["W3"].shape)
    print(NNet._param_dict["W1"].shape)
    print(NNet._param_dict["W2"].shape)
    print(NNet._param_dict["W3"].shape)
    assert len(grad_dict)==6
    assert grad_dict["W2"].shape[0]==NNet._param_dict["W2"].shape[0]
    assert grad_dict["W2"].shape[1]==NNet._param_dict["W2"].shape[1]

def test_predict():
    NNet=nn.NeuralNetwork(
        nn_arch=[{'input_dim':3,'output_dim':2,'activation':"relu"},{'input_dim':2,'output_dim':3,'activation':"sigmoid"},{'input_dim':3,'output_dim':1,'activation':"relu"}],
        lr=0.001,
        seed=37,
        batch_size=2,
        epochs=10,
        loss_function="MSE"
    )
    X_train=np.array([[1,2,3],[4,7,3],[0,5,-1]])
    y_hat=NNet.predict(X_train.T)
    y_train=np.array([[6,14,4]])
    assert y_hat.shape==y_train.shape

def test_binary_cross_entropy():
    NNet=nn.NeuralNetwork(
        nn_arch=[{'input_dim':3,'output_dim':2,'activation':"relu"},{'input_dim':2,'output_dim':3,'activation':"sigmoid"},{'input_dim':3,'output_dim':1,'activation':"relu"}],
        lr=0.001,
        seed=37,
        batch_size=2,
        epochs=10,
        loss_function="BCE"
    )
    y_hat=np.array([[0.1,0.6,0.2]])
    y=np.array([[0,1,0]])
    BCE=0.40
    BCE_NN=NNet._binary_cross_entropy(y,y_hat)
    assert round(BCE_NN,2) == BCE

def test_binary_cross_entropy_backprop():
    NNet=nn.NeuralNetwork(
        nn_arch=[{'input_dim':3,'output_dim':2,'activation':"relu"},{'input_dim':2,'output_dim':3,'activation':"sigmoid"},{'input_dim':3,'output_dim':1,'activation':"relu"}],
        lr=0.001,
        seed=37,
        batch_size=2,
        epochs=10,
        loss_function="BCE"
    )
    X_train=np.array([[7,4,8],[0,2,1],[-2,6,1]])
    y_train=np.array([[0,1,1]]).T
    X_val=np.array([[5,4,3],[0,0,5]])
    y_val=np.array([[0,1]]).T
    (train_error,val_error)=NNet.fit(X_train,y_train,X_val,y_val)
    assert train_error[0] > train_error[-1]

def test_mean_squared_error():
    NNet=nn.NeuralNetwork(
        nn_arch=[{'input_dim':3,'output_dim':2,'activation':"relu"},{'input_dim':2,'output_dim':3,'activation':"sigmoid"},{'input_dim':3,'output_dim':1,'activation':"relu"}],
        lr=0.001,
        seed=37,
        batch_size=2,
        epochs=10,
        loss_function="MSE"
    )
    y_hat=np.array([[1.1,3.5,9]])
    y=np.array([[2,4,8]])
    MSE=0.69
    MSE_NN=NNet._mean_squared_error(y,y_hat)
    assert round(MSE_NN,2) == MSE

def test_mean_squared_error_backprop():
    NNet=nn.NeuralNetwork(
        nn_arch=[{'input_dim':3,'output_dim':2,'activation':"relu"},{'input_dim':2,'output_dim':3,'activation':"sigmoid"},{'input_dim':3,'output_dim':1,'activation':"relu"}],
        lr=0.0001,
        seed=18,
        batch_size=2,
        epochs=10,
        loss_function="MSE"
    )
    X_train=np.array([[7,4,8],[0,2,1],[-2,6,1]])
    y_train=np.array([[0,1,1]]).T
    X_val=np.array([[5,4,3],[0,0,5]])
    y_val=np.array([[0,1]]).T
    (train_error,val_error)=NNet.fit(X_train,y_train,X_val,y_val)
    assert train_error[0] > train_error[-1]

def test_sample_seqs():
    seqs=["ACTG","GTAGA","ACAGTATA","GAT","GACACATATA"]
    bools=[True,False,False,False,False]
    [balanced_seqs,balanced_bools]=preprocess.sample_seqs(seqs,bools)
    assert len(balanced_seqs)==4
    assert sum(balanced_bools)==len(balanced_bools)/2
    

def test_one_hot_encode_seqs():
    seqs=["ACTG","GTAGA","ACAGTATA","GAT","GACACATATA"]
    bools=[True,False,False,False,False]
    [balanced_seqs,balanced_bools]=preprocess.sample_seqs(seqs,bools)
    seqs_one_hot=preprocess.one_hot_encode_seqs(balanced_seqs)
    assert seqs_one_hot.shape[0]==len(balanced_seqs)
    assert seqs_one_hot.shape[1]==40