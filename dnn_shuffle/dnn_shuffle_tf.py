import os
import math
import tensorflow as tf
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes



def shuffle_transform(img : tf.Tensor, block_size : int, key) -> tf.Tensor:
    """
    Divides input tensor image into blocks. Shuffles each block using key.

    Parameters:
    -----------------
    img (type: tf.Tensor, dim: batch_size x height x width x num_channels)
        image to shuffle
    block_size (type: int)
        size of each block
    key (type: bytes)
        bytes to initialise random number generator
    """

    assert(key is not None and block_size is not None and img is not None)

    # Setup seed and batch dim
    seed = gen_seed(img, key)
    batched = (len(img.shape) == 4)

    # Shuffle tensor using seeded random number generator
    block = tensor_to_blocks(img, batched, block_size)
    shuffled = shuffle_block(block, seed)

    return tf.reshape(shuffled, img.shape)
    


def gen_seed(tensor : tf.Tensor, key : bytes) -> int:
    '''
    Generates seed from tensor using key

    Parameters:
    -----------------
    tensor (type: tf.Tensor, dim: anything non-empty)
        tensor to generate seed from
    key (type: bytes)
        bytes to initialise random number generator

    Returns:
    -----------------
    None, but saves seed (type: int) as a class attribute
        seed generated from tensor and key
    '''

    # Convert tensor to bytes
    hashed = hashlib.sha3_256(tensor.numpy().tobytes())
    
    # Setup AES encryption
    init_vector = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(init_vector))
    encryptor = cipher.encryptor()
    
    # Generate seed using AES encryption
    cyphertext = encryptor.update(hashed.digest()) + encryptor.finalize()
    seed = int.from_bytes(cyphertext[::2][:4], byteorder="big")

    return seed



def gen_key(key_256 : bool=True):
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



def tensor_to_blocks(raw: tf.Tensor, batched : bool, block_size : int) -> tf.Tensor:
    '''
    Divides tensor into blocks of size block_size
    
    Parameters:
    -----------------
    raw (type: tf.Tensor, dim: batch_size x num_channels x height x width) 
        tensor to divide into blocks
    batched (type: bool)
        Whether the input has a batch dimension at the start. 
    block_size (type: int)
        size of each block

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
    num_blocks = math.ceil(raw.shape[-1] / block_size) * \
        math.ceil(raw.shape[-2] / block_size)
    num_channels = raw.shape[1] if batched else raw.shape[0]
    
    with tf.device(raw.device):
        blocks = tf.zeros(batch_dim, num_blocks, block_size**2 * num_channels)

        # Number batches and blocks
        for batch in range(batch_dim):
            for block in range(num_blocks):
                
                # Step through each block in height and width dimensions 
                for h_step in range(0, raw.shape[-2], block_size):
                    for w_step in range(0, raw.shape[-1], block_size):
                        
                        # Save flattened block
                        if batched:
                            highdim = raw[batch, :, h_step : h_step + block_size, 
                                    w_step : w_step + block_size]
                        else: 
                            highdim = raw[:, h_step : h_step + block_size, 
                                    w_step : w_step + block_size]
                            
                        blocks[batch, block] = tf.flatten(highdim)
    
    return blocks

def shuffle_block(raw : tf.Tensor, seed : int) -> tf.Tensor:
    '''
    Randomly shuffles each block in the input tensor. 

    Parameters:
    -----------------
    raw (type: tf.Tensor, dim: batch_size x num_blocks x (block_size^2 x num_channels))
        Tensor split into blocks.
    seed (type: int)
        Seed for random number generator

    Returns:
    -----------------
    shuffled (type: tf.Tensor, dim: batch_size x num_blocks x (block_size^2 x num_channels))
        Shuffled tensor. 
    '''

    for batch in range(raw.shape[0]):
        for block in range(raw.shape[1]):
            # Shuffle each block
            raw[batch, block, :] = tf.random.shuffle(
                tf.flatten(raw[batch, block, :]), seed=seed
            )
            
    if raw.shape[0] == 1:
        raw = tf.squeeze(raw, axis=0)
            
    return raw