# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    n=len(seqs)
    positives=[seqs[i] for i in range(n) if labels[i]==True]
    negatives=set(seqs)-set(positives)
    n_sample=int(n*0.8)
    pos_idx=np.random.randint(0,len(positives),n_sample)
    neg_idx=np.random.randint(0,len(negatives),n_sample)
    sample_pos=positives[pos_idx]
    sample_neg=positives[neg_idx]
    sample=sample_pos+sample_neg
    sample_bools=[True]*n_sample+[False]*n_sample
    return[sample, sample_bools]
    
    

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    base_onehot={"A":[1,0,0,0],
                 "T":[0,1,0,0],
                 "C":[0,0,1,0],
                 "G":[0,0,0,1]}
    encoding=[]
    for base in seq_arr:
        encoding.extend(base_onehot[base.upper()])
    return encoding