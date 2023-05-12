"""DEEP LOCK

This script encrypts the parameters of an input Pytorch model.

The following dependencies are required: `os`, `tensorflow`, `numpy`, `pickle`, `random`, `datetime`, `argparse`, and `struct`. 
"""

import os
import pickle
import random
import struct
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

random.seed(datetime.now().timestamp())




##############################################
#                                            #
#                  Byte Utils                #
#                                            #
##############################################

def float_to_bytes(num : float) -> bytes:
    return struct.pack('f', num)


def bytes_to_float(b : bytes) -> float:
    # Returns the float at the first element of the tuple
    return struct.unpack('f', b)[0]


def xor_bytes(bytes1 : bytes, bytes2 : bytes) -> bytes:
    ''' Returns new bytes object after XORing each byte '''
    return bytes([b1 ^ b2 for b1, b2 in zip(bytes1, bytes2)])


def rotate_bytes(byte : bytes) -> bytes:
    ''' Returns new bytes object after rotating each byte '''
    return byte[1:] + byte[:1]

def print_bytes(byte : bytes) -> None:
    ''' Prints a bytes object as a byte string '''

    out = ''
    for b in byte:
        # Get rid of 0b prefix and pad start with zeros
        string = bin(b)[2:]
        string = '0' * (8 - len(string)) + string
        # Add a space every 8 bits
        out += string + ' '
    
    print(out)




##############################################
#                                            #
#               Encryption Utils             #
#                                            #
##############################################

def update_round_const(prev_const : int) -> int:
    ''' Returns new round constant (integer) based on last round '''
    
    # Keep doubling 
    update = prev_const << 1
    if update < 256:
        return update
    # unless constant > 1 byte.
    else: 
        return update ^ 0x11b


def sub_bytes(byte : bytes, inverse : bool = False) -> bytes:
    ''' 
    Performs AES sub-bytes on each byte 
    
    Parameters
    ----------------
    byte (type: bytearray)
    - Bytes object to perform AES sub-bytes on
    inverse (type: boolean)
    - When true, runs byte through inverse rijndael s-box

    Returns
    ----------------
    bytes
    - Bytes object after AES sub-bytes
    '''

    # Create dict of s-box values and inverse s-box values
    sbox = {
        '00': '63', '01': '7c', '02': '77', '03': '7b', '04': 'f2', '05': '6b', '06': '6f', '07': 'c5', '08': '30', '09': '01', '0a': '67', '0b': '2b', '0c': 'fe', '0d': 'd7', '0e': 'ab', '0f': '76', 
        '10': 'ca', '11': '82', '12': 'c9', '13': '7d', '14': 'fa', '15': '59', '16': '47', '17': 'f0', '18': 'ad', '19': 'd4', '1a': 'a2', '1b': 'af', '1c': '9c', '1d': 'a4', '1e': '72', '1f': 'c0', 
        '20': 'b7', '21': 'fd', '22': '93', '23': '26', '24': '36', '25': '3f', '26': 'f7', '27': 'cc', '28': '34', '29': 'a5', '2a': 'e5', '2b': 'f1', '2c': '71', '2d': 'd8', '2e': '31', '2f': '15', 
        '30': '04', '31': 'c7', '32': '23', '33': 'c3', '34': '18', '35': '96', '36': '05', '37': '9a', '38': '07', '39': '12', '3a': '80', '3b': 'e2', '3c': 'eb', '3d': '27', '3e': 'b2', '3f': '75', 
        '40': '09', '41': '83', '42': '2c', '43': '1a', '44': '1b', '45': '6e', '46': '5a', '47': 'a0', '48': '52', '49': '3b', '4a': 'd6', '4b': 'b3', '4c': '29', '4d': 'e3', '4e': '2f', '4f': '84', 
        '50': '53', '51': 'd1', '52': '00', '53': 'ed', '54': '20', '55': 'fc', '56': 'b1', '57': '5b', '58': '6a', '59': 'cb', '5a': 'be', '5b': '39', '5c': '4a', '5d': '4c', '5e': '58', '5f': 'cf', 
        '60': 'd0', '61': 'ef', '62': 'aa', '63': 'fb', '64': '43', '65': '4d', '66': '33', '67': '85', '68': '45', '69': 'f9', '6a': '02', '6b': '7f', '6c': '50', '6d': '3c', '6e': '9f', '6f': 'a8', 
        '70': '51', '71': 'a3', '72': '40', '73': '8f', '74': '92', '75': '9d', '76': '38', '77': 'f5', '78': 'bc', '79': 'b6', '7a': 'da', '7b': '21', '7c': '10', '7d': 'ff', '7e': 'f3', '7f': 'd2', 
        '80': 'cd', '81': '0c', '82': '13', '83': 'ec', '84': '5f', '85': '97', '86': '44', '87': '17', '88': 'c4', '89': 'a7', '8a': '7e', '8b': '3d', '8c': '64', '8d': '5d', '8e': '19', '8f': '73', 
        '90': '60', '91': '81', '92': '4f', '93': 'dc', '94': '22', '95': '2a', '96': '90', '97': '88', '98': '46', '99': 'ee', '9a': 'b8', '9b': '14', '9c': 'de', '9d': '5e', '9e': '0b', '9f': 'db', 
        'a0': 'e0', 'a1': '32', 'a2': '3a', 'a3': '0a', 'a4': '49', 'a5': '06', 'a6': '24', 'a7': '5c', 'a8': 'c2', 'a9': 'd3', 'aa': 'ac', 'ab': '62', 'ac': '91', 'ad': '95', 'ae': 'e4', 'af': '79', 
        'b0': 'e7', 'b1': 'c8', 'b2': '37', 'b3': '6d', 'b4': '8d', 'b5': 'd5', 'b6': '4e', 'b7': 'a9', 'b8': '6c', 'b9': '56', 'ba': 'f4', 'bb': 'ea', 'bc': '65', 'bd': '7a', 'be': 'ae', 'bf': '08', 
        'c0': 'ba', 'c1': '78', 'c2': '25', 'c3': '2e', 'c4': '1c', 'c5': 'a6', 'c6': 'b4', 'c7': 'c6', 'c8': 'e8', 'c9': 'dd', 'ca': '74', 'cb': '1f', 'cc': '4b', 'cd': 'bd', 'ce': '8b', 'cf': '8a', 
        'd0': '70', 'd1': '3e', 'd2': 'b5', 'd3': '66', 'd4': '48', 'd5': '03', 'd6': 'f6', 'd7': '0e', 'd8': '61', 'd9': '35', 'da': '57', 'db': 'b9', 'dc': '86', 'dd': 'c1', 'de': '1d', 'df': '9e', 
        'e0': 'e1', 'e1': 'f8', 'e2': '98', 'e3': '11', 'e4': '69', 'e5': 'd9', 'e6': '8e', 'e7': '94', 'e8': '9b', 'e9': '1e', 'ea': '87', 'eb': 'e9', 'ec': 'ce', 'ed': '55', 'ee': '28', 'ef': 'df', 
        'f0': '8c', 'f1': 'a1', 'f2': '89', 'f3': '0d', 'f4': 'bf', 'f5': 'e6', 'f6': '42', 'f7': '68', 'f8': '41', 'f9': '99', 'fa': '2d', 'fb': '0f', 'fc': 'b0', 'fd': '54', 'fe': 'bb', 'ff': '16'
    }

    sbox_inverse = {
        '00': '52', '01': '09', '02': '6a', '03': 'd5', '04': '30', '05': '36', '06': 'a5', '07': '38', '08': 'bf', '09': '40', '0a': 'a3', '0b': '9e', '0c': '81', '0d': 'f3', '0e': 'd7', '0f': 'fb', 
        '10': '7c', '11': 'e3', '12': '39', '13': '82', '14': '9b', '15': '2f', '16': 'ff', '17': '87', '18': '34', '19': '8e', '1a': '43', '1b': '44', '1c': 'c4', '1d': 'de', '1e': 'e9', '1f': 'cb', 
        '20': '54', '21': '7b', '22': '94', '23': '32', '24': 'a6', '25': 'c2', '26': '23', '27': '3d', '28': 'ee', '29': '4c', '2a': '95', '2b': '0b', '2c': '42', '2d': 'fa', '2e': 'c3', '2f': '4e', 
        '30': '08', '31': '2e', '32': 'a1', '33': '66', '34': '28', '35': 'd9', '36': '24', '37': 'b2', '38': '76', '39': '5b', '3a': 'a2', '3b': '49', '3c': '6d', '3d': '8b', '3e': 'd1', '3f': '25', 
        '40': '72', '41': 'f8', '42': 'f6', '43': '64', '44': '86', '45': '68', '46': '98', '47': '16', '48': 'd4', '49': 'a4', '4a': '5c', '4b': 'cc', '4c': '5d', '4d': '65', '4e': 'b6', '4f': '92', 
        '50': '6c', '51': '70', '52': '48', '53': '50', '54': 'fd', '55': 'ed', '56': 'b9', '57': 'da', '58': '5e', '59': '15', '5a': '46', '5b': '57', '5c': 'a7', '5d': '8d', '5e': '9d', '5f': '84', 
        '60': '90', '61': 'd8', '62': 'ab', '63': '00', '64': '8c', '65': 'bc', '66': 'd3', '67': '0a', '68': 'f7', '69': 'e4', '6a': '58', '6b': '05', '6c': 'b8', '6d': 'b3', '6e': '45', '6f': '06', 
        '70': 'd0', '71': '2c', '72': '1e', '73': '8f', '74': 'ca', '75': '3f', '76': '0f', '77': '02', '78': 'c1', '79': 'af', '7a': 'bd', '7b': '03', '7c': '01', '7d': '13', '7e': '8a', '7f': '6b', 
        '80': '3a', '81': '91', '82': '11', '83': '41', '84': '4f', '85': '67', '86': 'dc', '87': 'ea', '88': '97', '89': 'f2', '8a': 'cf', '8b': 'ce', '8c': 'f0', '8d': 'b4', '8e': 'e6', '8f': '73', 
        '90': '96', '91': 'ac', '92': '74', '93': '22', '94': 'e7', '95': 'ad', '96': '35', '97': '85', '98': 'e2', '99': 'f9', '9a': '37', '9b': 'e8', '9c': '1c', '9d': '75', '9e': 'df', '9f': '6e', 
        'a0': '47', 'a1': 'f1', 'a2': '1a', 'a3': '71', 'a4': '1d', 'a5': '29', 'a6': 'c5', 'a7': '89', 'a8': '6f', 'a9': 'b7', 'aa': '62', 'ab': '0e', 'ac': 'aa', 'ad': '18', 'ae': 'be', 'af': '1b', 
        'b0': 'fc', 'b1': '56', 'b2': '3e', 'b3': '4b', 'b4': 'c6', 'b5': 'd2', 'b6': '79', 'b7': '20', 'b8': '9a', 'b9': 'db', 'ba': 'c0', 'bb': 'fe', 'bc': '78', 'bd': 'cd', 'be': '5a', 'bf': 'f4', 
        'c0': '1f', 'c1': 'dd', 'c2': 'a8', 'c3': '33', 'c4': '88', 'c5': '07', 'c6': 'c7', 'c7': '31', 'c8': 'b1', 'c9': '12', 'ca': '10', 'cb': '59', 'cc': '27', 'cd': '80', 'ce': 'ec', 'cf': '5f', 
        'd0': '60', 'd1': '51', 'd2': '7f', 'd3': 'a9', 'd4': '19', 'd5': 'b5', 'd6': '4a', 'd7': '0d', 'd8': '2d', 'd9': 'e5', 'da': '7a', 'db': '9f', 'dc': '93', 'dd': 'c9', 'de': '9c', 'df': 'ef', 
        'e0': 'a0', 'e1': 'e0', 'e2': '3b', 'e3': '4d', 'e4': 'ae', 'e5': '2a', 'e6': 'f5', 'e7': 'b0', 'e8': 'c8', 'e9': 'eb', 'ea': 'bb', 'eb': '3c', 'ec': '83', 'ed': '53', 'ee': '99', 'ef': '61', 
        'f0': '17', 'f1': '2b', 'f2': '04', 'f3': '7e', 'f4': 'ba', 'f5': '77', 'f6': 'd6', 'f7': '26', 'f8': 'e1', 'f9': '69', 'fa': '14', 'fb': '63', 'fc': '55', 'fd': '21', 'fe': '0c', 'ff': '7d'
    }
    
    # select sbox and make bytes mutable
    sb = sbox_inverse if inverse else sbox

    for i in range(len(byte)):
        # extract and pad hex code
        code = hex(byte[i])[2:]
        new_val = sb[ '0' * (2 - len(code)) + code ]
        # replace original byte with new 
        byte[i] = int(new_val, 16)
    
    return byte


def get_key(current_state : list = None, round_const : int = 1) -> tuple:
    ''' 
    Generates a 128 or 256 bit AES key for current round

    Parameters
    ----------------
    current_state (type: list of bytearrays)
    - The current key grouped by 32 bit vectors
    round_const (type: integer)
    - The round constant for the last round

    Returns
    ----------------
    current_state (type: list of bytearrays)
    - The next key grouped by 32 bit vectors
    round_const (type: integer)
    - The round constant for the current round
    '''

    # rotate, sub, and add constant to last vector
    transformed = rotate_bytes(current_state[-1])    
    transformed = sub_bytes(transformed)
    transformed[0] ^= round_const

    # xor vectors until end of round
    for i in range(len(current_state) - 1):
        current_state[i] = xor_bytes(current_state[i], transformed)
        transformed = current_state[i]

    # update round constant
    round_const = update_round_const(round_const)

    return current_state, round_const

def encrypt_model(model : tf.Module, key_256 : bool) -> bytes:
    '''
    Generates a key and uses it to encrypt model parameters.

    Parameters
    ----------------
    model (type: tf.Module or derivative class)
    - The model to encrypt
    key_256 (type: boolean)
    - Whether to use a 128 or 256 bit key

    Returns
    ----------------
    master_key (type: bytes)
    - The key you'll need to decrypt the model
    '''
    
    # Each vec has 32 bits to get 128 or 256 bit key
    num_vecs = 8 if key_256 else 4
    master_key = bytearray()
    key = []

    # Initialise master key
    for i in range(num_vecs):
        vector = bytearray()
        for j in range(4):
            vector.append(random.randint(0, 255)) 

        # Save key bytes       
        key.append(vector)
        master_key += vector

    # Get first round key to start encryption
    key, round_const = get_key(key)

    # Go through model weights
    for i, param in enumerate(model.variables()):
        if (i % 5 == 0): print(f'Layer {i}')
        # Convert parameter tensor to bytes
        param_bytes = bytearray(param.numpy().tobytes())

        # Encrypt 4 floats or doubles at a time
        for i in range(0, len(param_bytes), num_vecs * 4):
            param_bytes[i:i + num_vecs * 4] = sub_bytes(bytearray(
                xor_bytes(param_bytes[i:i + num_vecs * 4], b''.join(key))
            ))

            # Update key
            key, round_const = get_key(key, round_const)

        # Update parameter
        dtype = np.float32 if param.data.dtype == tf.float32 else np.float64
        param.assign(tf.convert_to_tensor(
            np.frombuffer(param_bytes, dtype=dtype).reshape(param.shape)
        ))
    
    return bytes(master_key)

def decrypt_model(model : tf.Module, key : bytes) -> None:
    '''
    Decrypts model using provided key

    Parameters
    ----------------
    model (type: tf.Module or derivative class)
    - The model to decrypt
    key (type: bytes)
    - The key to decrypt the model

    Returns
    ----------------
    None
    '''

    # Init round key and constant
    key = bytearray(key)
    key = [key[i:i + 4] for i in range(0, len(key), 4)]
    key, round_const = get_key(key)

    # Go through model weights
    for i, param in enumerate(model.variables()):
        if (i % 5 == 0): print(f'Layer {i}')
        # Convert parameter tensor to bytes
        param_bytes = bytearray(param.numpy().tobytes())

        # Decrypt 4 floats or doubles at a time
        for i in range(0, len(param_bytes), len(key) * 4):
            param_bytes[i:i + len(key) * 4] = xor_bytes(
                 b''.join(key), 
                sub_bytes(param_bytes[i:i + len(key) * 4], inverse = True) 
            )

            # Update key
            key, round_const = get_key(key, round_const)

        # Update parameter
        dtype = np.float32 if param.data.dtype == tf.float32 else np.float64
        param.assign(tf.convert_to_tensor(
            np.frombuffer(param_bytes, dtype=dtype).reshape(param.shape)
        ))




##############################################
#                                            #
#                 Model Utils                #
#                                            #
##############################################

def get_model(model_path : str) -> tf.Module:
    '''
    Load model, turn off redundant gradients.
    '''

    model = tf.keras.models.load_model(model_path)
    return model


def save_model(model : tf.Module, filepath : str) -> None: 
    ''' 
    Saves model to filesystem to load later. 
    
    Parameters
    ---------------------
    model (type: tf.Module or derivative class)
    - Model to save. 
    filepath (type: str)
    - NOTE: this must include a parent directory and empty subdirectory. 
    - Ex: saved_models/your_model - ex: your_model is empty.

    Returns
    ---------------------
    None
    '''

    # check directory
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    model.save(filepath)


def main(args : dict, decryption_mode : bool = False) -> None:
    
    # Set up model and secret key
    print("Fetching model and secret key")
    model = get_model(args.model)
    master_key = pickle.load(
        open(args.decryption_key, "rb")) if decryption_mode else None

    # Run decryption
    if decryption_mode:
        print("Starting decryption")
        decrypt_model(model, master_key)

        print("Saving model")
        out = 'decrypted_model.pt' 
        if (not args.output_model == 'encrypted_model.pt'):
            out = args.output_model
        save_model(model, out)

        print("Program successfully finished")
        return

    # Run encryption
    print("Starting encryption")
    master_key = encrypt_model(model, args.key_256)

    # Save key and model
    print("Saving encrypted model and key")
    save_model(model, args.output_model)
    pickle.dump(master_key, open(args.output_key, "wb"))

    print("Program successfully finished")


if __name__ == '__main__':
    # initialise argument parser
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('model', type=str, 
                        help='Filepath for Tensorflow model to encrypt/decrypt')

    parser.add_argument('--decrypt-mode', action='store_true', help="Decrypt model with key. Set --decryption-key alongside.")
    parser.add_argument('--key-256', action='store_true', help="Use 256-bit key, not 128-bit")

    parser.add_argument('--output-model',  type=str, default="saved_model/encrypted_model", 
                        help='Filepath to save model after encryption. Must include parent directory and empty subdirectory.')
    parser.add_argument('--output-key', type=str, default="decryption_key.pkl", 
                        help='Filepath to save secret key for decryption')
    parser.add_argument('--decryption-key', type=str, default="decryption_key.pkl", 
                        help='Filepath to load secret key for decryption. Set --decrypt-mode alongside.')
    
    # load arguments
    args = parser.parse_args()

    if not args.decrypt_mode and args.model:
        main(args)
    elif args.decrypt_mode and args.model and args.decryption_key:
        main(args, decryption_mode=True)
    else:
        print("Please provide a model filepath to encrypt. Or provide a model filepath, set --decrypt_mode, and provide a decryption key filepath to decrypt. Run --help for more details.")