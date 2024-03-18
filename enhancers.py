from nn import nn,io,preprocess
import numpy as np
import sklearn
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

rap1_seqs=io.read_text_file("./data/rap1-lieb-positives.txt")
negative_seqs=io.read_fasta_file("./data/yeast-upstream-1k-negative.fa")

all_seqs=rap1_seqs+negative_seqs
all_bools=[True]*len(rap1_seqs) + [False]*len(negative_seqs)

[seqs, bools]=preprocess.sample_seqs(all_seqs, all_bools)
seqs_one_hot=preprocess.one_hot_encode_seqs(seqs)

X=seqs_one_hot
y=np.array(bools)

# Sample 80/20 train/test split
train_idx=np.random.randint(0,len(X)-1,int(len(X)*0.8))
train_X=X[train_idx,:]
train_y=y[train_idx].reshape(train_X.shape[0],1)
test_idx=list(set(range(len(X)))-set(train_idx))
test_X=X[test_idx,:]
test_y=y[test_idx].reshape(test_X.shape[0],1)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#Initialize NN
NNet=nn.NeuralNetwork(
    nn_arch=[{'input_dim':X.shape[1],'output_dim':128,'activation':"sigmoid"},{'input_dim':128,'output_dim':64,'activation':"relu"},{'input_dim':64,'output_dim':1,"activation":"sigmoid"}],
    lr=0.0001,
    seed=37,
    batch_size=200,
    epochs=50,
    loss_function="BCE"
)

# Train NN
(train_error,val_error)=NNet.fit(train_X,train_y,test_X,test_y)
print(train_error)