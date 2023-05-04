"""ADVERSARIAL PARAMETER ENCRYPTION

This script encrypts the parameters of an input Pytorch model.

The following dependencies are required: `json`, `tensorflow`, `random`, `pickle`, `argparse`, `numpy`, and  `datetime`. 
"""

import json
import pickle
import random
import argparse
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
    with tf.device(device):
        encrypt_imgs = tf.constant(np.load(data_path), dtype=tf.float32)
        encrypt_labels = tf.constant(np.load(label_path), dtype=tf.int32)

    return encrypt_imgs, encrypt_labels


def get_model(model_path: str, device: str) -> tf.Module:
    '''
    Load model, turn off redundant gradient computation, and move to selected device
    '''

    model = tf.saved_model.load(model_path)
    model.trainable = True

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
    model (type: tf.Module - or a derivative class of this)
    - The model from which to draw layers from
    Returns
    --------------------
    list
    - Has names of the selected layers
    '''

    # Arrange layers into list
    random.seed(datetime.now().timestamp())
    model.trainable = True
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
        
        l_max = float(tf.math.reduce_max(layer_params))
        l_min = float(tf.math.reduce_min(layer_params))
        l_range = self.boundary_distance * (l_max - l_min)

        return l_min+l_range, l_max-l_range
    
    def compute_loss_grad(self, param_name):
        '''
        Computes average loss and gradient across minibatches
        
        Parameters
        -----------------------
        param_name (type: str)
        - Has an identifier for the trianable variable to be updated. 
        
        Returns
        -----------------------
        avg_loss (type: tf.Tensor, dim: 0, dtype: float)
        - Average loss across all batches
        avg_grad (type: tf.Tensor, dim: variable, dtype: float)
        - Average gradient across all batches. 
        '''
        
        # prep variables
        avg_loss = 0
        batches = 0
        avg_grad = None
        
        trainable_var = None
        for var in self.model.trainable_variables:
            if var.name == param_name:
                trainable_var = [var]
                break
        
        
        # go through batches
        for x, y in self.encrypt_data:
            print('.', end='')
            batches += 1
            
            # Compute loss
            with tf.GradientTape() as tape:
                pred = self.model(x)
                # Unlike pytorch, returns one loss value per tensor in batch
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred, from_logits=True)
            avg_loss += tf.math.reduce_mean(loss) # Averages loss across batch
            
            # Compute gradient
            dloss_dparams = tape.gradient(loss, trainable_var)            
            if avg_grad == None: 
                avg_grad = dloss_dparams[0]
            else:
                avg_grad += dloss_dparams[0]

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
        
        grad = grad * mask
        val = float(tf.reduce_max(grad))
        # returns flattened index for multidimensional tensors
        i = tf.argmax(tf.reshape(grad, [-1]))

        return val, int(i)
    
    def compute_update(self, param, grad, param_index):
        '''
        Computes update step for selected parameter
        
        Parameters
        --------------------
        param (type: tf.Tensor, dim: variable)
        - Layer housing selected parameter
        grad (type: float)
        - Gradient component for selected parameter
        param_index (type: int)
        - Index of selected parameter within the layer
        
        Returns
        --------------------
        update (type: float)
        -  new value for selected parameter
        unrolled_i (type: tuple)
        - the index of the element to update
        '''
        
        # compute update step
        param_range = tf.math.reduce_max(param) - tf.math.reduce_min(param)
        step = self.step_size * int(grad > 0) * param_range

        # get new param val
        unrolled_i = self.unroll_index(param_index, param.shape)
        # https://www.tensorflow.org/guide/tensor_slicing
        new_val = tf.slice(param, begin=unrolled_i, size=[1 for i in unrolled_i]) + step
        return new_val, unrolled_i
    
    
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
        tuple
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
        
        return tuple(out_i)
                        
    def encrypt_parameters(self):
        '''
        Makes selectively-targeted adversarial modifications to parameters.

        Parameters
        --------------------
        None

        Returns: 
        ---------------------
        dict
        - has keys for each encrypted layer and values indicating their 
        adjusted weights
        '''
        
        # Init variables and settings
        modifications = {}
        for layer in self.model.layers: 
            layer.trainable = False

        # Go through each model layer
        for layer in self.model.layers:
            param = None
            
            # Check if layer has variables to encrypt
            for v in layer.variables:
                if v.name in self.encrypt_layers:
                    layer.trainable = True
                    param = v
                    break
            
            # Skip to next layer if no layers to encrypt
            if (type(param) == type(None)): 
                continue
            print(param.name)
            
            # Generate mask to track which params in each layer are unusable
            modifications[param.name] = []
            mask = tf.ones(param.numpy().shape)
            bound_low, bound_high = self.compute_bounds(param.read_value())
            
            # Only run updates on current layer up to max iters
            for i in range(self.max_per_layer):
                print(i, end="")
                avg_loss, avg_grad = self.compute_loss_grad(param.name)

                # Abort if loss raised sufficiently
                if avg_loss > self.loss_threshold:
                    return modifications

                # Get adversarial update
                val, i = self.get_max_grad(mask, avg_grad)
                new_val, unrolled_i = self.compute_update(param.read_value(), val, i)
                
                # Mask current parameter if its gradients are too large
                if float(new_val) < bound_low or float(new_val) > bound_high:
                    # https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update
                    mask = tf.tensor_scatter_nd_update(mask, [unrolled_i], tf.constant([0.0]))
                
                # Save diff between old value and new value
                clipped = float(tf.clip_by_value(new_val, clip_value_min=bound_low, clip_value_max=bound_high))
                current = float(tf.slice(param.read_value(), begin=unrolled_i, size=[1 for i in unrolled_i]))
                modifications[param.name].append((unrolled_i, clipped - current))
                
                # Update old value
                updated_params = param.read_value()
                tf.tensor_scatter_nd_update(updated_params, [unrolled_i], tf.constant([clipped]))
                param.assign(updated_params)
            
            # Disable grad after layer operations finished
            layer.trainable = False
            
        return modifications
    

##################################################
############### DECRYPT KEY UTILS ################
##################################################

def decrypt_parameters(model : tf.Module, modifications : dict) -> None:
    '''
    Decrypts parameters using secret key
    
    Parameters
    --------------------
    model (type: tf.Module - or a derivative class)
    - The model containing parameters to decrypt
    modifications (type: dict)
    - An object recording the modifications made to parameters by layer.
    
    Returns
    --------------------
    None
    '''
    
    encrypted_layers = modifications.keys()
    
    # Select model layer
    for param in model.variables:
        if param.name in encrypted_layers:
            print(param.name)
            
            # Replace modified parameter
            for index, value in modifications[param.name]:
                updated_params = param.read_value()
                current = float(tf.slice(updated_params, begin=index, size=[1 for i in index]))
                tf.tensor_scatter_nd_update(updated_params, [index], tf.constant([current - value]))
                param.assign(updated_params)

def secret_formatter(
    secret : dict, 
    out_file : str, 
    format : str='pickle',
    load : bool=False) -> None:

    '''
    Converts secret dictionary into a more convenient format

    Parameters
    --------------------
    secret (type: dict)
    - An object recording the modifications made to parameters by layer.
    format (type: string)
    - Valid values are: 'json' or 'pickle'
    out_file (type: string)
    - The output file path to save to
    load (type: bool)
    - Whether to load a formatted secret instead of saving a raw secret.

    Returns
    --------------------
    None
    '''
    
    # file open mode
    mode = 'r' if load else 'w'
    if format == 'pickle':
        mode += 'b'
    
    with open(out_file, mode) as f:
        # reading file
        if load:
            if (format == 'json'):
                return json.load(f)
            elif (format == 'pickle'):
                return pickle.load(f)
            
        # writing file
        if (format == 'json'):
            json.dump(secret, f)
        elif (format == 'pickle'):
            pickle.dump(secret, f)


##################################################
##################### MAIN #######################
##################################################

def main(args):
    # Deal with decryption (simpler) first
    format = 'json' if args.json_key else 'pickle'

    if args.decrypt_mode:
        print("Decrypt mode enabled.")
        print("Extracting model and secret key")
        model = get_model(args.model, args.device)
        secret = secret_formatter({}, args.output_key, format, True)

        print("Decrypting model and saving")
        decrypt_parameters(model, secret)
        model_scripted = torch.jit.script(model)

        out_file = "decrypted_model.pt"
        if not args.output_model == "encrypted_model.tf":
            out_file = args.output_model
        model_scripted.save(out_file)

        print("Program successfully finished")
        return


    # Set up data, model, and layers
    print("Extracting model and encryption data")
    model = get_model(args.model, args.device)
    encrypt_layers = get_layer_set(args.max_layers, model)
    encrypt_imgs, encrypt_labels = get_data(args.data, args.labels, args.device)

    # Init hyperparameters
    print("Building dataset")
    dataset = tf.data.Dataset.from_tensor_slices((encrypt_imgs, encrypt_labels))
    encrypt_data = dataset.batch(64)

    x,y = next(iter(encrypt_data))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y, model(x), from_logits=True)
    loss = tf.math.reduce_mean(loss)
    print("Loss before encryption:", loss)
    loss_threshold = loss.item() * args.loss_multiple

    # Run encryption
    print("Starting encryption")
    instance = EncryptionUtils(model, encrypt_data, encrypt_layers, args.max_params, 
                               loss_threshold, args.step_size, 
                               args.boundary_distance, args.device)
    secret = instance.encrypt_parameters()

    x,y = next(iter(encrypt_data))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y, model(x), from_logits=True)
    loss = tf.math.reduce_mean(loss)
    print("Encrypted loss: ", loss)
    
    # Save secret and model
    print("Saving encrypted model")
    model_scripted = torch.jit.script(model)
    model_scripted.save(args.output_model)
    secret_formatter(secret, args.output_key, format)

    print("Program successfully finished")


if __name__ == '__main__':
    # initialise argument parser
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('data', type=str, 
                        help='Filepath for encryption data (or secret key in decrypt mode)')
    parser.add_argument('labels', type=str, 
                        help='Filepath for encryption labels (or location to save decrypted model in decrypt mode)')
    parser.add_argument('model', type=str, 
                        help='Filepath for tensorflow model to encrypt/decrypt')

    parser.add_argument('--disable-gpu', action='store_true', help='Disable GPU use')
    parser.add_argument('--json-key', action='store_true', help='Save secret key as JSON')
    parser.add_argument('--decrypt-mode', action='store_true', help="Decrypt model with key")

    parser.add_argument('--output-model',  type=str, default="encrypted_model.tf", 
                        help='Filepath to save model after encryption')
    parser.add_argument('--output-key', type=str, default="decryption_key.pkl", 
                        help='Filepath to save secret key for decryption')

    parser.add_argument('-l', '--max-layers', type=int, default=25,
                        help='Default 25. Maximum number of layers to encrypt')
    parser.add_argument('-b', '--batch-size', type=int, default=32, 
                        help="Default 32. Number of examples to process at once")
    parser.add_argument('-p', '--max-params', type=int, default=25,
                        help='Default 25. Maximum number of parameters to encrypt per layer')
    parser.add_argument('-d', '--boundary-distance', type=float, default=0.1,
                        help='Default 0.1. Set from 0 to 0.5. 0 = extreme where encrypted parameter values can be anywhere in range of existing parameter values. 0.5 = extreme where encrypted parameter values can only be at the midpoint of the range of existing parameter values.')
    parser.add_argument('-s', '--step-size', type=float, default=0.1,
                        help='Default 0.1. Step size per gradient-based update')
    parser.add_argument('-m', '--loss-multiple', type=float, default=5,
                        help='Default 5. Stops training when the loss has grown by N times')

    # load arguments
    args = parser.parse_args()
    args.device = None

    if not args.disable_gpu and \
    tf.config.list_physical_devices(device_type='GPU').length:
        args.device = 'gpu'
    else:
        args.device = 'cpu'
    
    if args.data and args.labels and args.model:
        main(args)