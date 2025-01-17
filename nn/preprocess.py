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
    negatives=list(set(seqs)-set(positives))
    n_sample=int(n/2)
    pos_idx=np.random.randint(0,len(positives),n_sample)
    neg_idx=np.random.randint(0,len(negatives),n_sample)
    sample_pos=[positives[x] for x in pos_idx]
    sample_neg=[negatives[x] for x in neg_idx]
    sample_unshuffled=sample_pos+sample_neg
    sample_bools_unshuffled=[True]*n_sample+[False]*n_sample
    shuffle_idx=np.arange(len(sample_unshuffled))
    np.random.shuffle(shuffle_idx)
    sample=[sample_unshuffled[x] for x in shuffle_idx]
    sample_bools=[sample_bools_unshuffled[x] for x in shuffle_idx]
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
    longest_seq=max([len(s) for s in seq_arr])
    encoding=np.zeros((len(seq_arr),longest_seq*4))
    # encodings_l=[]
    for k,seq in enumerate(seq_arr):
        encoding_single=[]
        for i,base in enumerate(seq):
            encoding_single.extend(base_onehot[base.upper()])
        encoding[k,0:4*(i+1)]=encoding_single
        #encodings_l.append(encoding_single)
    #encoding=np.array(encodings_l,dtype=object)
    return encoding