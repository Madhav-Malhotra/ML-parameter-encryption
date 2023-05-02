import random
import numpy as np
import tensorflow as tf
from datetime import datetime



##################################################
################# DATA LOADING ###################
##################################################

def get_data(data_path: str, label_path: str, device: str) -> list:
    '''
    Returns dataset for encryption (data, labels)

    Parameters
    ------------------
    data_path (type: string)
    - A filepath to a numpy array or pytorch tensor with processed images.
    label_path (type: string)
    - A filepath to a numpy array or pytorch tensor with image labels.
    device (type: str)
    - Where to store the data, can be 'cpu' or 'gpu'.

    Returns
    ------------------
    encrypt_imgs (type: tf.Tensor, dim: num_imgs x num_channels x height x width)
    - Preprocessed images to use for encryption
    encrypt_labels (type: tf.Tensor, dim: num_imgs)
    - Labels for above images
    '''

    # Load saved tensors
    encrypt_imgs = tf.convert_to_tensor(np.load(data_path), dtype=tf.float32)
    encrypt_labels = tf.convert_to_tensor(np.load(label_path), dtype=tf.int32)

    # Send tensors to specified device
    if device == 'gpu':
        encrypt_imgs = encrypt_imgs.gpu()
        encrypt_labels = encrypt_labels.gpu()
    else:
        encrypt_imgs = encrypt_imgs.cpu()
        encrypt_labels = encrypt_labels.cpu()

    return encrypt_imgs, encrypt_labels


def get_model(model_path: str, device: str) -> tf.Module:
    '''
    Load model, turn off redundant gradient computation, and move to selected device
    '''

    model = tf.saved_model.load(model_path)
    model.trainable = False

    if device == 'gpu':
        model = model.gpu()
    else:
        model = model.cpu()

    return model




##################################################
############### ENCRYPTION UTILS #################
##################################################

def get_layer_set(max_layers: int, model: tf.Module) -> list:
    '''
    Randomly selects model layers to choose parameters for encryption

    Parameters
    --------------------
    max_layers (type: int)
    - The maximum number of layers to be selected in the encryption set
    model (type: tf.Module)
    - The model from which to draw layers from

    Returns
    --------------------
    list
    - Has names of the selected layers
    '''

    # Arrange layers into list
    random.seed(datetime.now().timestamp())
    selected_layers = [layer.name for layer in model.trainable_variables]

    # Prune layers randomly if too many selected
    while len(selected_layers) > max_layers:
        rand_i = random.randint(0, len(selected_layers) - 1)
        del selected_layers[rand_i]

    return selected_layers


class EncryptionUtils:
    def __init__(
        self, 
        model : tf.Module, 
        encrypt_data : tf.data.Dataset, 
        encrypt_layers : list, 
        max_per_layer : int, 
        loss_threshold : float, 
        step_size : float, 
        boundary_distance : float,
        device : str):
        
        self.model = model
        self.encrypt_data = encrypt_data
        self.encrypt_layers = encrypt_layers
        self.max_per_layer = max_per_layer
        self.loss_threshold = loss_threshold
        self.step_size = step_size
        self.boundary_distance = boundary_distance
        self.device = device
    
    def compute_bounds(self, layer_params : tf.Tensor) -> list:
        '''
        Computes boundaries for parameter values in selected layer

        Parameters
        --------------------
        layer_params (type: tf.Tensor, dim: variable)
        - Weights (multidimensional) or biases (1D) of current layer.

        Returns
        --------------------
        bound_low (type: float)
        - lower limit on parameter values
        bound_high (type: float)
        - upper limit on parameter values
        '''
        
        with tf.no_gradient():
            l_max = tf.math.reduce_max(layer_params)
            l_min = tf.math.reduce_min(layer_params)
            l_range = self.boundary_distance * (l_max - l_min)
        
        return l_min+l_range, l_max-l_range
    
    def compute_avg_loss(self):
        '''
        Computes average loss and gradient across minibatches

        Parameters
        -----------------------
        None

        Returns
        -----------------------
        avg_loss (type: tf.Tensor, dim: 0, dtype: float)
        - Average loss across all batches
        avg_grad (type: tf.Tensor, dim: variable, dtype: float)
        - Average gradient across all batches. 
        '''

        avg_loss = 0
        batches = 0
        avg_grad = None
        
        for x, y in self.encrypt_data:
            batches += 1
            
            with tf.GradientTape() as tape:
                pred = self.model(x)
                # Unlike pytorch, returns one loss value per tensor in batch
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred, from_logits=True)
            
            avg_loss += tf.math.reduce_mean(loss) # Averages loss across batch
            dloss_dparams = tape.gradient(loss, self.model.trainable_variables)
            
            if avg_grad == None: 
                avg_grad = dloss_dparams
            else:
                avg_grad += dloss_dparams

        return avg_loss / batches, avg_grad / batches

    def get_max_grad(self, mask, grad):
        '''
        Finds the maximum gradient vector component
        
        Parameters
        --------------------
        mask (type: tf.Tensor, dim: variable)
        - A tensor of 1s/0s for each parameter in current layer 
        - 1 indicates a parameter is updatable. 
        - Has same dim as current layer
        grad (type: tf.Tensor, dim: variable)
        - Gradient of current layer
        
        Returns
        --------------------
        val (type: float)
        - Maximum gradient component for current layer's params
        i (type: int)
        - Max gradient component's index for current layer's params
        - Layers with multi-dimensional parameters are flattened
          before index is computed.
        '''
        
        with tf.no_grad():
            # Return the maximum gradient val/index
            val = float(tf.reduce_max(grad * mask))
            # returns flattened index for multidimensional tensors
            i = tf.math.argmax(grad)

            return val, int(i)
        
    def unroll_index(self, i, new_dim):
        '''
        Turns a one-dimensional index into a new dimensional shape
        
        Parameters
        --------------------
        i (type: int)
        - index of flattened tensor
        new_dim (type: tuple)
        - dimensions to adjust the flattened index into
        
        Returns
        --------------------
        list
        - 1+ elements representing indices along different dimensions
        '''
        
        # Handle exception for low-dimensional index
        if len(new_dim) == 1:
            return [i]
        elif not len(new_dim):
            return [0]
        
        # Create useful variables
        out_i = []
        new_dim = list(new_dim)
        prod = int(new_dim[0])
        for j in range(1, len(new_dim)):
            prod *= int(new_dim[j])
        
        while (len(new_dim)):
            # Get current index
            prod /= int(new_dim[0])
            out_i.append(int(i // prod))
            
            # Prepare for next round
            i = int(i % prod)
            del new_dim[0]
        
        return out_i
    
    def compute_update(self, param, grad, param_index):
        '''
        Computes update step for selected parameter
        
        Parameters
        --------------------
        param (type: tf.Tensor)
        - Layer housing selected parameter
        grad (type: float)
        - Gradient component for selected parameter
        param_index (type: int)
        - Index of selected parameter within the layer
        
        Returns
        --------------------
        update (type: float)
        -  new value for selected parameter
        unrolled_i (type: list)
        - the index of the element to update
        '''
        
        # compute update step
        param_range = tf.math.reduce_max(param) - tf.math.reduce_max(param)
        step = self.step_size * int(grad > 0) * param_range

        # get new param val
        shape = tuple(tf.shape(param).numpy().tolist())
        unrolled_i = self.unroll_index(param_index, shape)
        size = [1 for i in unrolled_i]

        # https://www.tensorflow.org/guide/tensor_slicing
        new_val = tf.slice(param, begin=unrolled_i, size=size) + step
        return new_val, unrolled_i
    
    