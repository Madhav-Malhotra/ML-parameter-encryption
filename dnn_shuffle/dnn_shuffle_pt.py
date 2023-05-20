import math
import torch
import random
import hashlib

# Create a custom transform class that shuffles the data
class ShuffleTransform(object):
    """
    Divides input tensor image into blocks. Shuffles each block using key.

    Parameters:
    -----------------
    key (type: int) 
        key to use for shuffling
    block_size (type: int)
        size of each block
    """

    def __init__(self, key, block_size):
        self.key = key
        self.block_size = block_size
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        pass

    def gen_seed(self, tensor : torch.Tensor) -> int:
        '''
        Generates seed from tensor using key

        Parameters:
        -----------------
        tensor (type: torch.Tensor)
            tensor to generate seed from

        Returns:
        -----------------
        seed (type: int)
            seed generated from tensor and key
        '''

        # Convert tensor to bytes
        tensor_bytes = tensor.numpy().tobytes()

        # Generate seed from tensor_bytes and key (adjust this to AES the key and the hash output)
        seed = int(hashlib.sha3_256(tensor_bytes + self.key).hexdigest(), 16) % 10**8

        return seed

    def tensor_to_blocks(self, raw: torch.Tensor) -> torch.Tensor:
        '''
        Divides tensor into blocks of size block_size
        
        Parameters:
        -----------------
        raw (type: torch.Tensor, dim: batch_size x num_channels x height x width) 
            tensor to divide into blocks

        Returns:
        -----------------
        blocks (type: torch.Tensor, dim: batch_size x num_blocks x (block_size^2 x num_channels))
            Contains raw values, divided into blacks
        '''

        # For every block_size x block_size square in height x width
        # select and flatten values in all channels in that block
        num_blocks = math.ceil(raw.shape[-1] / self.block_size) * \
            math.ceil(raw.shape[-2] / self.block_size)
        blocks = torch.zeros(raw.shape[0], num_blocks, self.block_size**2 * raw.shape[1])

        for batch in range(raw.shape[0]):
            for block in range(num_blocks):
                for h_step in range(0, raw.shape[2], self.block_size):
                    for w_step in range(0, raw.shape[-1], self.block_size):
                        blocks[batch, block] = torch.flatten(
                            raw[batch, :, h_step + self.block_size, w_step + self.block_size]
                        )
        
        print(blocks.shape)