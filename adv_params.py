##################################################
########### LIBRARIES AND ARGUMENTS ##############
##################################################

import json
import torch 
import random
import pickle
import argparse
import numpy as np
from datetime import datetime
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

# initialise argument parser
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-gpu', action='store_true', help='Disable GPU use')

parser.add_argument('--data', help='Filepath for encryption data')
parser.add_argument('--labels', help='Filepath for encryption labels')
parser.add_argument('--model', help='Filepath for pickled model to encrypt')
parser.add_argument('--output-model', help='Filepath for encrypted model')
parser.add_argument('--output-key', help='Filepath for decryption key')

parser.add_argument('--max-layers', help='Maximum number of layers to encrypt')
parser.add_argument('--max-weights', help='Maximum number of weights to encrypt per layer')
parser.add_argument('--boundary-distance', help='0-1 decimal percent on how far from extremes encrypted parameter values should be')
parser.add_argument('--step-size', help='Step size per gradient-based update')
parser.add_argument('--max-weights', help='Maximum number of weights to encrypt per layer')



# load arguments
args = parser.parse_args()
args.device = None

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')




##################################################
################# DATA LOADING ###################
##################################################

def get_data(data_path : str, label_path : str, numpy_load : bool=True) -> list:
    '''
    Returns dataset for encryption (data, labels)
    
    Parameters
    ------------------
    data_path (type: string)
    - A filepath to a numpy array or pytorch tensor with processed images.
    label_path (type: string)
    - A filepath to a numpy array or pytorch tensor with image labels.
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
        encrypt_imgs = torch.from_numpy(np.load(data_path)).to(args.device)
        encrypt_labels = torch.from_numpy(np.load(label_path)).to(args.device)
    else: 
        encrypt_imgs = torch.load(data_path, map_location=args.device)
        encrypt_labels = torch.load(label_path, map_location=args.device)
        
    return encrypt_imgs, encrypt_labels


def get_model(model_path : str) -> torch.nn.Module:
    '''
    Load model, turn off redundant gradients, and move to selected device
    '''

    model = torch.load(model_path)
    model.train(False)
    model.to(args.device)
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
        boundary_distance : float):
        
        self.model = model
        self.encrypt_data = encrypt_data
        self.encrypt_layers = encrypt_layers
        self.max_per_layer = max_per_layer
        self.loss_threshold = loss_threshold
        self.step_size = step_size
        self.boundary_distance = boundary_distance
    
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
            print('.', end="")
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
            mask = torch.ones(param.shape, dtype=bool, device=args.device).detach()
            bound_low, bound_high = self.compute_bounds(param)

            # Only run updates on current layer up to max iters
            for i in range(self.max_per_layer):
                print(i, end="")
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

def decrypt_parameters(model : torch.nn.Module, modifications : dict) -> None:
    '''
    Decrypts parameters using secret key
    
    Parameters
    --------------------
    model (type: torch.nn.Module)
    - The model containing parameters to decrypt
    modifications (type: dict)
    - An object recording the modifications made to parameters by layer.
    
    Returns
    --------------------
    None
    '''
    
    encrypted_layers = modifications.keys()
    
    # Select model layer
    for name, param in model.named_parameters():
        if name in encrypted_layers:
            print(name)
            
            # Replace modified parameter
            for index, value in modifications[name]:
                with torch.no_grad():
                    param[index] = param[index] - value


def modifications_formatter(
    modifications : dict, 
    out_file : str, 
    format : str='pickle') -> None:

    '''
    Converts modifications dictionary into a more convenient format

    Parameters
    --------------------
    modifications (type: dict)
    - An object recording the modifications made to parameters by layer.
    format (type: string)
    - Valid values are: 'json' or 'pickle'
    out_file (type: string)
    - The output file path to save to

    Returns
    --------------------
    None
    '''
    
    write_mode = 'wb' if format == 'pickle' else 'w'
    
    with open(out_file, write_mode) as f:
        if (format == 'json'):
            json.dump(modfications, f)
        elif (format == 'pickle'):
            pickle.dump(modifications, f)


##################################################
##################### MAIN #######################
##################################################

def main():
    # Set up data, model, and layers
    encrypt_imgs, encrypt_labels = get_data(args.data, args.labels)
    model = get_model(args.model)
    encrypt_layers = get_layer_set(args.max_layers, model)

    # Init hyperparameters
    dataset = TensorDataset(encrypt_imgs, encrypt_labels)
    encrypt_data = DataLoader(dataset, batch_size=args.batch_size)

    x,y = next(iter(encrypt_data))
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(model(x), y.long())
        print("Loss before encryption:", loss)
    loss_threshold = loss.item() * 5

    # Run encryption
    instance = EncryptionUtils(model, encrypt_data, encrypt_layers, args.max_weights, loss_threshold, args.step_size, args.boundary_distance)
    modifications = instance.encrypt_parameters()

    x,y = next(iter(encrypt_data))
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(model(x), y.long())
        print("Encrypted loss: ", loss)
    
    # Save modifications and model
    torch.save(model, args.output_model)
    modifications_formatter(modifications, args.output_key)

if __name__ == '__main__':
    main()