import os
import math
import torch
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


# Create a custom transform class that shuffles the data
class ShuffleTransform():
    """
    Divides input tensor image into blocks. Shuffles each block using key.

    Parameters:
    -----------------
    key (type: int) 
        key to use for shuffling
    block_size (type: int)
        size of each block
    """

    def __init__(self, block_size, key = None):
        if (key):
            self.key = key
        else: 
            self.key = self.gen_key()
        self.block_size = block_size
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        
        # Setup seed and batch dim
        self.gen_seed(img)
        batched = (len(img.shape) == 4)

        # Shuffle tensor using seeded random number generator
        block = self.tensor_to_blocks(img, batched)
        shuffled = self.shuffle_block(block, batched)

        return torch.reshape(shuffled, img.shape)
    
    def gen_key(self, key_256 : bool=True):
        '''
        Initialises secret key to process data. 
        
        Parameters
        -----------------
        key_256 (type: bool)
        - Whether to generate a 256-bit key (not 128 bit key)
        - Use 256 bit keys for SHA-256/SHA3-256 hashes
        
        Returns
        -----------------
        master_key (type: bytearray)
        - randomly initialised key 
        '''
        
        num_bytes = 32 if key_256 else 16
        master_key = bytearray(os.urandom(num_bytes))
        
        return master_key
        

    def gen_seed(self, tensor : torch.Tensor) -> int:
        '''
        Generates seed from tensor using key

        Parameters:
        -----------------
        tensor (type: torch.Tensor)
            tensor to generate seed from

        Returns:
        -----------------
        None, but saves seed (type: int) as a class attribute
            seed generated from tensor and key
        '''

        # Convert tensor to bytes
        hashed = hashlib.sha3_256(tensor.numpy().tobytes())
        
        # Setup AES encryption
        init_vector = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(init_vector))
        encryptor = cipher.encryptor()
        
        # Generate seed using AES encryption
        cyphertext = encryptor.update(hashed.digest()) + encryptor.finalize()
        seed = int.from_bytes(cyphertext[::2][:4], byteorder="big")

        self.seed = seed

    def tensor_to_blocks(self, raw: torch.Tensor, batched : bool) -> torch.Tensor:
        '''
        Divides tensor into blocks of size block_size
        
        Parameters:
        -----------------
        raw (type: torch.Tensor, dim: batch_size x num_channels x height x width) 
            tensor to divide into blocks
        batched (type: bool)
        - Whether the input has a batch dimension at the start. 

        Returns:
        -----------------
        blocks (type: torch.Tensor, dim: batch_size x num_blocks x (block_size^2 x num_channels))
            Contains raw values, divided into blacks
        '''

        assert(len(raw.shape) == 4 or len(raw.shape) == 3)
        # Adjust for batched versus non batched input
        batch_dim = raw.shape[0] if batched else 1
        
        # For every block_size x block_size square in height x width
        # select and flatten values in all channels in that block
        num_blocks = math.ceil(raw.shape[-1] / self.block_size) * \
            math.ceil(raw.shape[-2] / self.block_size)
        num_channels = raw.shape[1] if batched else raw.shape[0]
        blocks = torch.zeros(batch_dim, num_blocks, self.block_size**2 * num_channels)

        # Number batches and blocks
        for batch in range(batch_dim):
            for block in range(num_blocks):
                
                # Step through each block in height and width dimensions 
                for h_step in range(0, raw.shape[-2], self.block_size):
                    for w_step in range(0, raw.shape[-1], self.block_size):
                        
                        # Save flattened block
                        if batched:
                            highdim = raw[batch, :, h_step : h_step + self.block_size, 
                                    w_step : w_step + self.block_size]
                        else: 
                            highdim = raw[:, h_step : h_step + self.block_size, 
                                    w_step : w_step + self.block_size]
                            
                        blocks[batch, block] = torch.flatten(highdim)
        
        return blocks

    def shuffle_block(self, raw : torch.Tensor, batched : bool) -> torch.Tensor:
        '''
        Randomly shuffles each block in the input tensor. 

        Parameters:
        -----------------
        raw (type: torch.Tensor, dim: batch_size x num_blocks x (block_size^2 x num_channels))
            Tensor split into blocks.
        batched (type: bool)
        - Whether the input has a batch dimension at the start. 

        Returns:
        -----------------
        shuffled (type: torch.Tensor, dim: batch_size x num_blocks x (block_size^2 x num_channels))
            Shuffled tensor. 
        '''

        # Setup random number generator
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        for batch in range(raw.size(0)):
            for block in range(raw.size(1)):
                # Shuffle each block
                idx = torch.randperm(raw.size(-1), generator=rng)
                raw[batch, block] = raw[batch, block][idx]
                
        if raw.size(0) == 1:
            raw = torch.squeeze(raw, 0)
                
        return raw