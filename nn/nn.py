# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        Z_curr=W_curr.dot(A_prev) + b_curr
        if activation=="sigmoid":
            A_curr=self._sigmoid(Z_curr) 
        elif activation == "relu":
            A_curr=self._relu(Z_curr)
        return A_curr, Z_curr #Should be correct

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """

        # Flatten inputs into the input layer - one-hot encoded?
        # For layer in nn:
        #     Give layer activation to single_forward
        #     Store A and Z
        cache={}
        input=X
        cache["A0"]=input
        for layer in range(len(self.arch)):
            layer=layer+1
            A,Z=self._single_forward(self._param_dict["W"+str(layer)],self._param_dict["b"+str(layer)],input,self.arch[layer-1]["activation"])
            cache["A"+str(layer)]=A
            cache["Z"+str(layer)]=Z
            input=A
        return A, cache
            
        # return output and full A, Z


    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix. Previous meaning i+1 layer
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        dA_prev=np.zeros(W_curr.shape)
        if activation_curr=="sigmoid":
            #dA_prev=W_curr.T.dot(self._sigmoid_backprop(dA_curr,Z_curr))
            dA_prev=W_curr.T.dot(dA_curr)*self._sigmoid_backprop(dA_curr,Z_curr)
        elif activation_curr=="relu":
            dA_prev=W_curr.T.dot(dA_curr)*self._relu_backprop(dA_curr,Z_curr)
        dW_curr=dA_prev.dot(A_prev.T)
        db_curr=np.sum(dA_prev,axis=1,keepdims=True)
        return dA_prev, dW_curr,db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        #initialize last layer
        grad_dict={}
        Z_curr=cache["Z"+str(len(self.arch))]
        if self._loss_func=="BCE":
            dA_curr=self._binary_cross_entropy_backprop(y,y_hat)
        elif self._loss_func=="MSE":
            dA_curr=self._mean_squared_error_backprop(y,y_hat)
        #print(f"Last layer dA_curr has shape {dA_curr.shape}\n")
        #dZ=self._sigmoid_backprop(dA_curr,Z_curr)
        grad_dict["W"+str(len(self.arch))]=dA_curr.dot(cache["A"+str(len(self.arch)-1)].T)
        grad_dict["b"+str(len(self.arch))]=np.sum(dA_curr,axis=1,keepdims=True)
        for i in range(1,len(self.arch)):         
            layer=len(self.arch)-i #should range from 1 to #layers -1
            W_curr=self._param_dict["W"+str(layer+1)]
            b_curr=self._param_dict["b"+str(layer+1)]
            activation_curr=self.arch[layer-1]["activation"]
            Z_curr=cache["Z"+str(layer)]
            A_prev=cache["A"+str(layer-1)] #we store X=A0 so this should be ok
            dA_prev, dW_curr,db_curr=self._single_backprop(W_curr,b_curr,Z_curr,A_prev,dA_curr,activation_curr)
            grad_dict["W"+str(layer)]=dW_curr
            grad_dict["b"+str(layer)]=db_curr
            dA_curr=dA_prev
            #print(f"Next layer dA_curr has shape {dA_curr.shape}")
        return grad_dict
            

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for layer in range(1,len(self.arch)+1):
            W_old=self._param_dict["W"+str(layer)]
            b_old=self._param_dict["b"+str(layer)]
            W_new=W_old-self._lr*grad_dict["W"+str(layer)]
            b_new=b_old-self._lr*grad_dict["b"+str(layer)]
            self._param_dict["W"+str(layer)]=W_new
            self._param_dict["b"+str(layer)]=b_new
            assert W_old.shape==W_new.shape
            assert b_old.shape==b_new.shape
            #assert not np.all(grad_dict["W"+str(layer)]==0)
            # print(f"Updated layer {layer} weights")

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        train_error=[]
        val_error=[]
        for iteration in range(self._epochs):
            batch_idx=np.random.randint(0,len(X_train)-1,self._batch_size)
            X_batch=X_train[batch_idx,:].T #Each column is an example. Each feature is a row
            y_batch=y_train[batch_idx].T #y is a row vector
            #
            y_hat_batch,cache=self.forward(X_batch)
            assert y_batch.shape==y_hat_batch.shape
            grad_dict=self.backprop(y_batch,y_hat_batch,cache)
            # for i,entry in enumerate(grad_dict):
            #     print(f"{entry} : {grad_dict[entry].shape}")
            self._update_params(grad_dict)
            y_hat_val=self.predict(X_val.T)
            y_hat_val=y_hat_val.T
            assert y_val.shape==y_hat_val.shape
            if self._loss_func=="BCE":
                train_error_entry=self._binary_cross_entropy(y_batch,y_hat_batch)
                val_error_entry=self._binary_cross_entropy(y_val,y_hat_val)
            elif self._loss_func=="MSE":
                train_error_entry=self._mean_squared_error(y_batch,y_hat_batch)
                val_error_entry=self._mean_squared_error(y_val,y_hat_val)
            train_error.append(train_error_entry)
            val_error.append(val_error_entry)
            print(f"Done with iteration {iteration}. Train error: {round(train_error_entry,2)} Val error: {round(val_error_entry,2)}")
        return (train_error,val_error)
            

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat,cache=self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        Z=np.clip(Z,a_min=-100,a_max=None)
        nl_transform=1/(1+np.exp(-Z))
        #np.array([1/(1+np.exp(z)) for z in Z])
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        A=self._sigmoid(Z)
        dZ=A * (1-A)
        # np.array([self._sigmoid(z)*(1-self._sigmoid(z)) for z in Z]) @ dA
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform=np.maximum(0,Z)
        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        #A=self._relu(Z)
        dZ=Z>0
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        n=len(y_hat)
        y_hat[y_hat==0]=1e-10
        y_hat[y_hat==1]=1-1e-10
        #loss=(-1/n)*sum(sum([yi*np.log(yhi)+(1-yi)*np.log(yhi) for yi,yhi in zip(y,y_hat)]))
        loss=-np.mean(y * np.log2(y_hat) + (1 - y) * np.log2(1 - y_hat))
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        n=len(y_hat)
        y_hat[y_hat==0]=1e-10
        y_hat[y_hat==1]=1-1e-10
        dA=(-1/n)*(y/y_hat - (1-y)/(1-y_hat))
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        n=len(y_hat)
        loss=(1/n)*sum(sum([(yi-yhi)**2 for yi,yhi in zip(y,y_hat)]))
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        n=len(y_hat)
        dA=(1/n)*(y-y_hat)
        #np.array([float(yi)-yhi for yi,yhi in zip(y,y_hat)])
        return dA