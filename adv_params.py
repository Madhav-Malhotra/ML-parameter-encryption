"""ADVERSARIAL PARAMETER ENCRYPTION

This script encrypts the parameters of an input Pytorch model.

The following dependencies are required: `json`, `torch`, `random`, `pickle`,
`argparse`, `numpy`, and  `datetime`.

This file can also be imported as a module with these objects:

    * EncryptionUtils: Class to encrypt model parameters.
    * get_layer_set: Randomly selects model parameters for encryption.
    * decrypt_parameters: Decrypts a model's parameters given a secret key.
    * secret_formatter: Saves the secret key in a JSON/pickle format. 
"""

import json
import torch 
import random
import pickle
import argparse
import numpy as np
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader




##################################################
################# DATA LOADING ###################
##################################################

def get_data(data_path : str, label_path : str, device : torch.device, numpy_load : bool=True) -> list:
    '''
    Returns dataset for encryption (data, labels)
    
    Parameters
    ------------------
    data_path (type: string)
    - A filepath to a numpy array or pytorch tensor with processed images.
    label_path (type: string)
    - A filepath to a numpy array or pytorch tensor with image labels.
    device (type: torch.device)
    - Where to store the data
    numpy_load (type: bool)
    - Whether to use numpy load (compared to pytorch load)
    
    Returns
    encrypt_imgs (type: torch.Tensor, dim: num_imgs x num_channels x height x width)
    - Preprocessed images to use for encryption
    encrypt_labels (type: torch.Tensor, dim: num_imgs)
    - Labels for above images
    '''
    
    # Load saved tensors
    if numpy_load:
        encrypt_imgs = torch.from_numpy(np.load(data_path)).to(device)
        encrypt_labels = torch.from_numpy(np.load(label_path)).to(device)
    else: 
        encrypt_imgs = torch.load(data_path, map_location=device)
        encrypt_labels = torch.load(label_path, map_location=device)
        
    return encrypt_imgs, encrypt_labels


def get_model(model_path : str, device : torch.device) -> torch.nn.Module:
    '''
    Load model, turn off redundant gradients, and move to selected device
    '''

    model = torch.jit.load(model_path)
    model.train(False)
    model.to(device)
    return model




##################################################
############### ENCRYPTION UTILS #################
##################################################

def get_layer_set(max_layers : int, model : torch.nn.Module) -> list:
    '''
    Randomly selects model layers to choose parameters for encryption
    
    Parameters
    --------------------
    max_layers (type: int)
    - The maximum number of layers to be selected in the encryption set
    model (type: torch.nn.Module)
    - The model from which to draw layers from
    
    Returns
    --------------------
    list
    - Has names of the selected layers
    '''
    
    # Arrange layers into list
    random.seed(datetime.now().timestamp())
    selected_layers = [name for name,_ in model.named_parameters()]

    # Prune layers randomly if too many selected
    while len(selected_layers) > max_layers:
        rand_i = random.randint(0, len(selected_layers) - 1)
        del selected_layers[rand_i]
    
    return selected_layers


class EncryptionUtils:
    def __init__(
        self, 
        model : torch.nn.Module, 
        encrypt_data : torch.utils.data.DataLoader, 
        encrypt_layers : list, 
        max_per_layer : int, 
        loss_threshold : float, 
        step_size : float, 
        boundary_distance : float,
        device : torch.device):
        
        self.model = model
        self.encrypt_data = encrypt_data
        self.encrypt_layers = encrypt_layers
        self.max_per_layer = max_per_layer
        self.loss_threshold = loss_threshold
        self.step_size = step_size
        self.boundary_distance = boundary_distance
        self.device = device
    
    def compute_bounds(self, layer_params):
        '''
        Computes boundaries for parameter values in selected layer
        
        Parameters
        --------------------
        layer_params (type: torch.Tensor, dim: variable)
        - Weights (multidimensional) or biases (1D) of current layer.
        
        Returns
        --------------------
        bound_low (type: float)
        - lower limit on parameter values
        bound_high (type: float)
        - upper limit on parameter values
        '''
        
        with torch.no_grad():
            l_max = torch.max(layer_params).detach()
            l_min = torch.min(layer_params).detach()
            l_range = self.boundary_distance * (l_max - l_min)
        
        return l_min+l_range, l_max-l_range
    
    def compute_avg_loss(self):
        '''
        Computes average loss and gradient across minibatches
        '''
        
        avg_loss = 0
        batches = 0

        # Compute gradient and average loss for batches
        for x,y in self.encrypt_data:
            batches += 1
            
            pred = self.model(x)
            loss = torch.nn.functional.cross_entropy(pred, y.long())
            
            avg_loss += loss.item()
            loss.backward()
        
        return avg_loss / batches, batches

    def get_max_grad(self, batches, mask, param):
        '''
        Finds the maximum gradient vector component
        
        Parameters
        --------------------
        batches (type: int)
        - Number of batches to average gradient over. 
        mask (type: torch.Tensor, dim: variable)
        - A tensor of 1s/0s for each parameter in current layer 
        - 1 indicates a parameter is updatable. 
        - Has same dim as current layer
        param (type: torch.nn.Parameter)
        - Layer parameters to update. 
        
        Returns
        --------------------
        val (type: float)
        - Maximum gradient component for current layer's params
        i (type: int)
        - Max gradient component's index for current layer's params
        - Layers with multi-dimensional parameters are flattened
          before index is computed.
        '''
        
        with torch.no_grad():
            # Return the maximum gradient val/index
            avg_grad = (param.grad / batches) * mask
            val = torch.max(avg_grad).item()
            # returns flattened index for multidimensional tensors
            i = torch.argmax(avg_grad)

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
        param (type: torch.nn.Parameter)
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
        param_range = torch.max(param).detach() - torch.min(param).detach()
        step = self.step_size * int(grad > 0) * param_range

        # get new param val
        unrolled_i = self.unroll_index(param_index, param.shape)
        new_val = param[tuple(unrolled_i)].detach() + step
        return new_val, unrolled_i
                

    def encrypt_parameters(self):
        '''
        Makes selectively-targeted adversarial modifications to parameters.

        Parameters
        --------------------
        encrypt_data (type: torch.utils.data.DataLoader)
        - generates batched images and labels to evaluate loss
        encrypt_layers (type: dict)
        - keys are layer names and values are model parameters to adjust
        max_per_layer (type: int)
        - the maximum number of parameters which can be modified per layer
        loss_threshold (type: float)
        - loss at which the encryption process stops

        Returns: 
        ---------------------
        dict
        - has keys for each encrypted layer and values indicating their 
        adjusted weights
        '''

        modifications = {}

        # Enable grad for layers to encrypt
        for name, param in self.model.named_parameters():
            if name not in self.encrypt_layers:
                continue
            else:
                param.requires_grad = True
            print(name)
            
            # Generate mask to track which params in each layer are unusable
            modifications[name] = []
            mask = torch.ones(param.shape, dtype=bool, device=self.device).detach()
            bound_low, bound_high = self.compute_bounds(param)

            # Only run updates on current layer up to max iters
            for i in range(self.max_per_layer):
                print(i)
                self.model.zero_grad()
                avg_loss, batches = self.compute_avg_loss()

                # Abort if loss raised sufficiently
                if avg_loss > self.loss_threshold:
                    return modifications

                # Get adversarial update
                val, i = self.get_max_grad(batches, mask, param)
                new_val, unrolled_i = self.compute_update(param, val, i)
                
                # Update mask and parameter
                if new_val < bound_low or new_val > bound_high:
                    mask[tuple(unrolled_i)] = False
                
                with torch.no_grad():
                    clipped = torch.clamp(new_val, min=bound_low, max=bound_high)
                    j = tuple(unrolled_i)
                    modifications[name].append((j, clipped - param[j]))
                    param[j] = clipped
            
            # Disable grad after layer operations finished
            param.requires_grad = False
            
        return modifications



##################################################
############### DECRYPT KEY UTILS ################
##################################################

def decrypt_parameters(model : torch.nn.Module, secret : dict) -> None:
    '''
    Decrypts parameters using secret key
    
    Parameters
    --------------------
    model (type: torch.nn.Module)
    - The model containing parameters to decrypt
    secret (type: dict)
    - An object recording the modifications made to parameters by layer.
    
    Returns
    --------------------
    None
    '''
    
    encrypted_layers = secret.keys()
    
    # Select model layer
    for name, param in model.named_parameters():
        if name in encrypted_layers:
            print(name)
            
            # Replace modified parameter
            for index, value in secret[name]:
                with torch.no_grad():
                    param[index] = param[index] - value


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
        if not args.output_model == "encrypted_model.pt":
            out_file = args.output_model
        model_scripted.save(out_file)

        print("Program successfully finished")
        return


    # Set up data, model, and layers
    print("Extracting model and encryption data")
    model = get_model(args.model, args.device)
    encrypt_layers = get_layer_set(args.max_layers, model)
    encrypt_imgs, encrypt_labels = get_data(args.data, args.labels, args.device,
                                            not args.pt_data)

    # Init hyperparameters
    print("Building dataset")
    dataset = TensorDataset(encrypt_imgs, encrypt_labels)
    encrypt_data = DataLoader(dataset, batch_size=args.batch_size)

    x,y = next(iter(encrypt_data))
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(model(x), y.long())
        print("Loss before encryption:", loss)
    loss_threshold = loss.item() * args.loss_multiple

    # Run encryption
    print("Starting encryption")
    instance = EncryptionUtils(model, encrypt_data, encrypt_layers, args.max_params, 
                               loss_threshold, args.step_size, 
                               args.boundary_distance, args.device)
    secret = instance.encrypt_parameters()

    x,y = next(iter(encrypt_data))
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(model(x), y.long())
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
                        help='Filepath for torchscript model to encrypt/decrypt')

    parser.add_argument('--disable-gpu', action='store_true', help='Disable GPU use')
    parser.add_argument('--json-key', action='store_true', help='Save secret key as JSON')
    parser.add_argument('--pt-data', action='store_true', 
                        help="Pytorch dataset files instead of numpy")
    parser.add_argument('--decrypt-mode', action='store_true', help="Decrypt model with key")

    parser.add_argument('--output-model',  type=str, default="encrypted_model.pt", 
                        help='Filepath to save model after encryption')
    parser.add_argument('--output-key', type=str, default="decryption_key.pkl", 
                        help='Filepath to save secret key for decryption')

    parser.add_argument('-l', '--max-layers', type=int, default=25,
                        help='Maximum number of layers to encrypt')
    parser.add_argument('-b', '--batch-size', type=int, default=32, 
                        help="Number of dataset images to process at once")
    parser.add_argument('-p', '--max-params', type=int, default=25,
                        help='Maximum number of parameters to encrypt per layer')
    parser.add_argument('-d', '--boundary-distance', type=float, default=0.1,
                        help='Set from 0 to 0.5. 0 = extreme where encrypted parameter values can be anywhere in range of existing parameter values. 0.5 = extreme where encrypted parameter values can only be at the midpoint of the range of existing parameter values.')
    parser.add_argument('-s', '--step-size', type=float, default=0.1,
                        help='Step size per gradient-based update')
    parser.add_argument('-m', '--loss-multiple', type=float, default=5,
                        help='Stops training when the loss has grown by N times')

    # load arguments
    args = parser.parse_args()
    args.device = None

    if not args.disable_gpu and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    
    if args.data and args.labels and args.model:
        main(args)