# ML Parameter Encryption

This repository has implementations of recent encryption algorithms for AI models. These algorithms aim to **prevent unauthorised users from accessing AI models.** My intention in creating these implementations is to make theoretical algorithms more production-ready and useful. 

The algorithms implemented are described in the following papers: 
1. A. Pyone, M. Maung, and H. Kiya, ‘Training DNN Model with Secret Key for Model Protection’, in 2020 IEEE 9th Global Conference on Consumer Electronics (GCCE), Kobe, Japan: IEEE, Oct. 2020, pp. 818–821. doi: [10.1109/GCCE50665.2020.9291813](https://doi.org/10.1109/GCCE50665.2020.9291813).
2. M. Alam, S. Saha, D. Mukhopadhyay, and S. Kundu, ‘Deep-Lock: Secure Authorization for Deep Neural Networks’. arXiv, Aug. 13, 2020. Accessed: Apr. 29, 2023. \[Online\]. Available: http://arxiv.org/abs/2008.05966
3. M. Xue, Z. Wu, J. Wang, Y. Zhang, and W. Liu, ‘AdvParams: An Active DNN Intellectual Property Protection Technique via Adversarial Perturbation Based Parameter Encryption’. arXiv, May 28, 2021. Accessed: Apr. 29, 2023. \[Online\]. Available: http://arxiv.org/abs/2105.13697




--------------------




## AdvParams

<details><summary><h3>Pytorch</h3></summary>

This algorithm currently has a Pytorch implementation for any model derived from the [`torch.jit.ScriptModule`](https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule) or [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class. 

Models can easily be converted between these two formats. Still, the **Module-based usage section is ideal for beginners** since it shows how to apply the core encryption functions to the common `torch.nn.Module` format.

[See this Kaggle notebook for a demo](https://www.kaggle.com/code/madhavmalhotra/advparams-parameter-encryption/notebook) of the AdvParam algorithm.




<details><summary><h3>Script-based Usage</h3></summary>

Download `/adv_params/adv_prams_pt.py` from this repository into some directory. In that same directory, **save your model** using code like the following: 

```python
import torch

# Example pretrained model. Replace with your own model.
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

# Convert torch.nn.Module to torch.jit.ScriptModule
model_scripted = torch.jit.script(model)

# Once you have a torch.jit.ScriptModule, save it
model_scripted.save('model_scripted.pt')
```

In addition, **save preprocessed data and associated labels** in that directory (in numpy or Pytorch pickled format). This data should be ready to directly load and input to your model's feedforward function. For an example, download the pretrained [MobileNetV2](https://paperswithcode.com/method/mobilenetv2) model and a subset of 1000 preprocessed images from the [ImageNet dataset](https://www.kaggle.com/c/imagenet-object-localization-challenge) from [this Kaggle dataset](https://kaggle.com/datasets/46f0aca63f7c255adb88e2608635096e2f423fcc7e7efb35d3c0180416f3a809).

**To encrypt the model, use the following command**. It specifies that the script should get encryption data from `encrypt_imgs.npy`, get encryption labels from `encrypt_labels.npy`, and load the model from `model_scripted.pt`. 
```bash
python3 adv_params_pt.py encrypt_imgs.npy encrypt_labels.npy model_scripted.pt
```

As an output, the script will save a secret key necessary to decrypt the model in `decryption_key.pkl`. Also, it will save the encrypted model to `encrypted_model.pt`. You can **decrypt the model** with this data using this command. It specifies that the script should get the decryption key and encrypted model from the above files, decrypt the model, and then save the decrypted model as `decrypted_model.pt`. 
```bash
python3 adv_params_pt.py decryption_key.pkl decrypted_model.pt encrypted_model.pt --decrypt-mode
```

To see other script options, run 
```
python3 adv_params_pt.py --help

usage: adv_params_pt.py [-h] [--disable-gpu] [--json-key] [--pt-data] [--decrypt-mode] [--output-model OUTPUT_MODEL] [--output-key OUTPUT_KEY] [-l MAX_LAYERS] [-b BATCH_SIZE]
                        [-p MAX_PARAMS] [-d BOUNDARY_DISTANCE] [-s STEP_SIZE] [-m LOSS_MULTIPLE]
                        data labels model

ADVERSARIAL PARAMETER ENCRYPTION This script encrypts the parameters of an input Pytorch model. The following dependencies must be installed: `json`, `torch`, `random`, `pickle`, `argparse`, `numpy`, and `datetime`.

positional arguments:
  data                  Filepath for encryption data (or secret key in decrypt mode)
  labels                Filepath for encryption labels (or location to save decrypted model in decrypt mode)
  model                 Filepath for torchscript model to encrypt/decrypt

options:
  -h, --help            show this help message and exit
  --disable-gpu         Disable GPU use
  --json-key            Save secret key as JSON
  --pt-data             Pytorch dataset files instead of numpy
  --decrypt-mode        Decrypt model with key
  --output-model OUTPUT_MODEL
                        Filepath to save model after encryption
  --output-key OUTPUT_KEY
                        Filepath to save secret key for decryption
  -l MAX_LAYERS, --max-layers MAX_LAYERS
                        Default 25. Maximum number of layers to encrypt
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Default 32. Number of examples to process at once
  -p MAX_PARAMS, --max-params MAX_PARAMS
                        Default 25. Maximum number of parameters to encrypt per layer
  -d BOUNDARY_DISTANCE, --boundary-distance BOUNDARY_DISTANCE
                        Default 0.1. Set from 0 to 0.5. 
                        0 = encrypted parameters can be anywhere in range of existing parameters. 
                        0.5 = encrypted parameters must be existing parameter range's midpoint.
  -s STEP_SIZE, --step-size STEP_SIZE
                        Default 0.1. Step size per gradient-based update
  -m LOSS_MULTIPLE, --loss-multiple LOSS_MULTIPLE
                        Default 5. Stops training when loss has grown by N times
```

Some notes explaining the above options: 
- The **positional arguments are different for encryption mode and decryption mode**. In encryption mode, the arguments in order are `data_source.npy label_source.npy model_source.pt`. In decryption mode, the arguments in order are `decryption_key.pkl output_model_filepath.pt model_to_decrypt.pt`.
- A key aim of the algorithm is to keep encrypted (modified) parameters indistinguishable from unencrypted (original) parameters. To do this, it keeps encrypted parameter values within certain boundaries set within the range of existing parameter values in each layer. These **boundaries are computed using the boundary distance ($\beta$) as follows**:
  - $B_{low} = \min{W_l} + \beta \cdot (\max{W_l} - \min{W_l})$ - where $B_{low}$ is the lowest acceptable encrypted parameter value and $W_l$ represents the parameters of the $lth$ layer. 
  -  $B_{high} = \max{W_l} - \beta \cdot (\max{W_l} - \min{W_l})$ - where $B_{high}$ is the highest acceptable encrypted parameter value.
- The loss multiple sets the algorithm to stop encryption early if the loss has been raised (performance has been deteriorated) sufficiently. Ex: If you set this value to 5, then parameters stop being modified (encrypted) when the average loss across batches is 5 times higher than the loss prior to encryption. 

</details>







<details><summary><h3>Module-based Usage</h3></summary>

You can also import the Python script in `adv_params/adv_params_pt.py` as a module in your own Python scripts. The following dependencies must be installed: `json`, `torch`, `random`, `pickle`, `argparse`, `numpy`, and `datetime`.

Useful objects to import are:
- `EncryptionUtils`: Class to encrypt model parameters. 
- `get_layer_set`: Randomly selects model parameters for encryption. 
- `decrypt_parameters`: Decrypts a model's parameters given a secret key. 
- `secret_formatter`: Saves the secret key in a JSON/pickle format.

Here is some example code **showing how models can be encrypted**. Put this file in the same directory as `adv_params_pt.py`
```python
import torch
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from adv_params_pt import EncryptionUtils, get_layer_set

# Get a torch.nn.Module or torch.jit.ScriptModule however you want
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

# Load your dataset however you want
dataset = TensorDataset(YOUR_DATA_HERE, YOUR_LABELS_HERE)
encrypt_data = DataLoader(dataset, batch_size=32)


# Declare hyperparameters
max_layers = 25                 # max layers to encrypt
max_params = 25                 # max parameters to encrypt per layer
step_size = 0.1                 # adjusts gradient update size
loss_multiple = 5               # stop encrypting when loss raised 5x
boundary_distance = 0.1         # set 0-0.5. See docs for details
device = torch.device('cpu')

# Set a max loss to stop at
x,y = next(iter(encrypt_data))
with torch.no_grad():
    loss = F.cross_entropy(model(x), y.long())
loss_threshold = loss.item() * loss_multiple


# Encrypt the model
encrypt_layers = get_layer_set(max_layers, model) 
instance = EncryptionUtils(model, encrypt_data, encrypt_layers,
                          max_params, loss_threshold, step_size, 
                          boundary_distance, device)
secret = instance.encrypt_parameters() 


# Save your secret (decryption key) and encrypted model parameters at the end.
torch.save(model, 'encrypted_model.pkl')
f = open('decrpytion_key.pkl', 'wb')
pickle.dump(secret, f)
f.close()
```

A **note about the boundary distance hyperparameter**:
- A key aim of the algorithm is to keep encrypted (modified) parameters indistinguishable from unencrypted (original) parameters. To do this, it keeps encrypted parameter values within certain boundaries set within the range of existing parameter values in each layer. These boundaries are computed using the boundary distance ($\beta$) as follows:
  - $B_{low} = \min{W_l} + \beta \cdot (\max{W_l} - \min{W_l})$ - where $B_{low}$ is the lowest acceptable encrypted parameter value and $W_l$ represents the parameters of the $lth$ layer. 
  -  $B_{high} = \max{W_l} - \beta \cdot (\max{W_l} - \min{W_l})$ - where $B_{high}$ is the highest acceptable encrypted parameter value.

Finally, this code snippet shows **how to decrypt models**:
```python
import torch
import pickle
from adv_params_pt import decrypt_parameters

# Load encrypted data
model = torch.load(model, 'encrypted_model.pkl')
secret = pickle.load('decryption_key.pkl')

# Decrypt parameters and save model
decrypt_parameters(model, secret)
torch.save(model, 'decrypted_model.pkl')
```

</details>

</details>











<details><summary><h3>Tensorflow</h3></summary>

This algorithm currently has a Tensorflow implementation for any model derived from the [`tf.Module`](https://www.tensorflow.org/api_docs/python/tf/Module) class. This includes both the [Functional](https://www.tensorflow.org/guide/keras/functional) and [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) API.  

[See this Kaggle notebook for a demo](https://www.kaggle.com/code/madhavmalhotra/tf-fork-of-advparams-parameter-encryption/notebook) of the AdvParam algorithm.






<details><summary><h3>Script-based Usage</h3></summary>

Download `/adv_params/adv_prams_tf.py` from this repository into some directory. In that same directory, **save your model** using code like the following: 
```python
import tensorflow as tf

# Example pretrained model. Replace with your own model.
model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    weights='imagenet',
    classifier_activation=None) # we want to get back model logits
model.trainable = True

# Save model
model.save('saved_model/raw_model')
```

In addition, **save preprocessed data and associated labels** in that directory (in numpy format). This data should be ready to directly load and input to your model's feedforward function. For an example, download a subset of 1000 preprocessed images from the [ImageNet dataset](https://www.kaggle.com/c/imagenet-object-localization-challenge) from [this Kaggle dataset](https://kaggle.com/datasets/46f0aca63f7c255adb88e2608635096e2f423fcc7e7efb35d3c0180416f3a809). You can use them with the MobileNetV2 model above. Just be sure to reshape the images to have the channel dimension last: 
```python
import tensorflow as tf
import numpy as np

encrypt_imgs = tf.constant(np.load('YOUR_DIR_HERE/encrypt_imgs.npy'), dtype=tf.float32)

# Need to reshape channels to be last dim for MobileNetV2
encrypt_imgs = tf.transpose(encrypt_imgs, perm=[0, 2, 3, 1])
```

**To encrypt the model, use the following command**. It specifies that the script should get encryption data from `encrypt_imgs.npy`, get encryption labels from `encrypt_labels.npy`, and load the model from `saved_model/raw_model`. 
```bash
python3 adv_params_tf.py encrypt_imgs.npy encrypt_labels.npy saved_model/raw_model
```

As an output, the script will save a secret key necessary to decrypt the model in `decryption_key.pkl`. Also, it will save the encrypted model to `saved_model/encrypted_model`. You can **decrypt the model** with this data using the next command. It specifies that the script should get the decryption key and encrypted model from the above files, decrypt the model, and then save the decrypted model as `saved_model/decrypted_model`. 
```bash
python3 adv_params_tf.py decryption_key.pkl saved_model/encrypted_model saved_model/decrypted_model --decrypt-mode
```

To see other script options, run 
```
python3 adv_params_tf.py --help

usage: temp.py [-h] [--disable-gpu] [--json-key] [--decrypt-mode] [--output-model OUTPUT_MODEL] [--output-key OUTPUT_KEY] [-l MAX_LAYERS] [-b BATCH_SIZE] [-p MAX_PARAMS] [-d BOUNDARY_DISTANCE] [-s STEP_SIZE] [-m LOSS_MULTIPLE] data labels model

ADVERSARIAL PARAMETER ENCRYPTION This script encrypts the parameters of an input Tensorflow model. The following dependencies are required: `os`, `json`, `tensorflow`, `random`, `pickle`, `argparse`, `numpy`, and `datetime`.

positional arguments:
  data                  Filepath for encryption data (or secret key in decrypt mode)
  labels                Filepath for encryption labels (or location to save decrypted model in decrypt mode)
  model                 Filepath for tensorflow model to encrypt/decrypt

options:
  -h, --help            show this help message and exit
  --disable-gpu         Disable GPU use
  --json-key            Save secret key as JSON
  --decrypt-mode        Decrypt model with key
  --output-model OUTPUT_MODEL
                        Default saved_model/encrypted_model. Filepath to save model after encryption. Must include parent directory and empty subdirectory.
  --output-key OUTPUT_KEY
                        Default decryption_key.pkl. Filepath to save secret key for decryption
  -l MAX_LAYERS, --max-layers MAX_LAYERS
                        Default 25. Maximum number of layers to encrypt
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Default 32. Number of examples to process at once
  -p MAX_PARAMS, --max-params MAX_PARAMS
                        Default 25. Maximum number of parameters to encrypt per layer
  -d BOUNDARY_DISTANCE, --boundary-distance BOUNDARY_DISTANCE
                        Default 0.1. Set from 0 to 0.5. 
                        0 = extreme where encrypted parameter values can be anywhere in range of existing parameter values. 
                        0.5 = extreme where encrypted parameter values can only be at the midpoint of the range of existing     
                        parameter values.
  -s STEP_SIZE, --step-size STEP_SIZE
                        Default 0.1. Step size per gradient-based update
  -m LOSS_MULTIPLE, --loss-multiple LOSS_MULTIPLE
                        Default 5. Stops training when the loss has grown by N times
```

Some notes explaining the above options: 
- The **positional arguments are different for encryption mode and decryption mode**. In encryption mode, the arguments in order are `data_source.npy label_source.npy model_dir`. In decryption mode, the arguments in order are `decryption_key.pkl output_model_dir model_to_decrypt_dir`.
- A key aim of the algorithm is to keep encrypted (modified) parameters indistinguishable from unencrypted (original) parameters. To do this, it keeps encrypted parameter values within certain boundaries set within the range of existing parameter values in each layer. These **boundaries are computed using the boundary distance ($\beta$) as follows**:
  - $B_{low} = \min{W_l} + \beta \cdot (\max{W_l} - \min{W_l})$ - where $B_{low}$ is the lowest acceptable encrypted parameter value and $W_l$ represents the parameters of the $lth$ layer. 
  -  $B_{high} = \max{W_l} - \beta \cdot (\max{W_l} - \min{W_l})$ - where $B_{high}$ is the highest acceptable encrypted parameter value.
- The loss multiple sets the algorithm to stop encryption early if the loss has been raised (performance has been deteriorated) sufficiently. Ex: If you set this value to 5, then parameters stop being modified (encrypted) when the average loss across batches is 5 times higher than the loss prior to encryption. 

</details>

<details><summary><h3>Module-based Usage</h3></summary>

You can also import the Python script in `adv_params/adv_params_tf.py` as a module in your own Python scripts. The following dependencies must be installed: `json`, `tensorflow`, `random`, `pickle`, `argparse`, `numpy`, and `datetime`.

Useful objects to import are:
- `EncryptionUtils`: Class to encrypt model parameters. 
- `get_layer_set`: Randomly selects model parameters for encryption. 
- `decrypt_parameters`: Decrypts a model's parameters given a secret key. 
- `secret_formatter`: Saves the secret key in a JSON/pickle format.

Here is some example code **showing how models can be encrypted**. Put this file in the same directory as `adv_params_tf.py`
```python
import os
import pickle
import tensorflow as tf
from adv_params_tf import EncryptionUtils, get_layer_set

# Get a tf.Module (or derivative class) however you want
model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    weights='imagenet',
    classifier_activation=None) 
model.trainable = True

# Load your dataset however you want
dataset = tf.data.Dataset.from_tensor_slices((YOUR_DATA_HERE, YOUR_LABELS_HERE))
encrypt_data = dataset.batch(32)


# Declare hyperparameters
max_layers = 25             # max layers to encrypt
max_params = 25             # max parameters to encrypt per layer
step_size = 0.1             # adjusts gradient update size
loss_multiple = 5           # stop encrypting when loss raised 5x
boundary_distance = 0.1     # set 0-0.5. See docs for details
device = 'cpu'

# Set a max loss to stop at
x,y = next(iter(encrypt_data))
loss = tf.keras.losses.sparse_categorical_crossentropy(y, model(x), from_logits=True)
loss = tf.math.reduce_mean(loss)
loss_threshold = float(loss) * loss_multiple


# Encrypt the model
encrypt_layers = get_layer_set(max_layers, model) 
instance = EncryptionUtils(model, encrypt_data, encrypt_layers,
                          max_params, loss_threshold, step_size, 
                          boundary_distance, device)
secret = instance.encrypt_parameters() 


# Save your encrypted model parameters
filepath = 'saved_model/encrypted_model'
if not os.path.exists(filepath):
    os.makedirs(filepath)
model.save(filepath)

# Save your secret (decryption key)
f = open('decrpytion_key.pkl', 'wb')
pickle.dump(secret, f)
f.close()
```

A **note about the boundary distance hyperparameter**:
- A key aim of the algorithm is to keep encrypted (modified) parameters indistinguishable from unencrypted (original) parameters. To do this, it keeps encrypted parameter values within certain boundaries set within the range of existing parameter values in each layer. These boundaries are computed using the boundary distance ($\beta$) as follows:
  - $B_{low} = \min{W_l} + \beta \cdot (\max{W_l} - \min{W_l})$ - where $B_{low}$ is the lowest acceptable encrypted parameter value and $W_l$ represents the parameters of the $lth$ layer. 
  -  $B_{high} = \max{W_l} - \beta \cdot (\max{W_l} - \min{W_l})$ - where $B_{high}$ is the highest acceptable encrypted parameter value.

Finally, this code snippet shows **how to decrypt models**:
```python
import pickle
import tensorflow as tf
from adv_params_tf import decrypt_parameters

# Load encrypted data
device = 'cpu'
with tf.device(device):
        model = tf.keras.models.load_model('saved_model/encrypted_model')
        model.trainable = True
secret = pickle.load('decryption_key.pkl')

# Decrypt parameters and save model
decrypt_parameters(model, secret)
model.save('saved_model/decrypted_model')
```

</details>

</details>












--------------------












## DeepLock

<details><summary><h3>Pytorch</h3></summary>

This algorithm currently has a Pytorch implementation for any model derived from the [`torch.jit.ScriptModule`](https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule) or [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class. 

Models can easily be converted between these two formats. Still, the **Module-based usage section is ideal for beginners** since it shows how to apply the core encryption functions to the common `torch.nn.Module` format.

[See this Kaggle notebook for a demo](https://www.kaggle.com/code/madhavmalhotra/deeplock-parameter-encryption) of the DeepLock algorithm.




<details><summary><h3>Script-based Usage</h3></summary>

Download `/deep_lock/deep_lock_pt.py` from this repository into some directory. In that same directory, **save your model** using code like the following: 

```python
import torch

# Example pretrained model. Replace with your own model.
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

# Convert torch.nn.Module to torch.jit.ScriptModule
model_scripted = torch.jit.script(model)

# Once you have a torch.jit.ScriptModule, save it
model_scripted.save('model_scripted.pt')
```

In addition, **save preprocessed data and associated labels** in that directory (in numpy or Pytorch pickled format). This data should be ready to directly load and input to your model's feedforward function. For an example, download the pretrained [MobileNetV2](https://paperswithcode.com/method/mobilenetv2) model and a subset of 1000 preprocessed images from the [ImageNet dataset](https://www.kaggle.com/c/imagenet-object-localization-challenge) from [this Kaggle dataset](https://kaggle.com/datasets/46f0aca63f7c255adb88e2608635096e2f423fcc7e7efb35d3c0180416f3a809).

**To encrypt the model, use the following command**. It specifies that the script should get encryption data from `encrypt_imgs.npy`, get encryption labels from `encrypt_labels.npy`, and load the model from `model_scripted.pt`. 
```bash
python3 adv_params_pt.py encrypt_imgs.npy encrypt_labels.npy model_scripted.pt
```

As an output, the script will save a secret key necessary to decrypt the model in `decryption_key.pkl`. Also, it will save the encrypted model to `encrypted_model.pt`. You can **decrypt the model** with this data using this command. It specifies that the script should get the decryption key and encrypted model from the above files, decrypt the model, and then save the decrypted model as `decrypted_model.pt`. 
```bash
python3 adv_params_pt.py decryption_key.pkl decrypted_model.pt encrypted_model.pt --decrypt-mode
```

To see other script options, run 
```
python3 adv_params_pt.py --help

usage: adv_params_pt.py [-h] [--disable-gpu] [--json-key] [--pt-data] [--decrypt-mode] [--output-model OUTPUT_MODEL] [--output-key OUTPUT_KEY] [-l MAX_LAYERS] [-b BATCH_SIZE]
                        [-p MAX_PARAMS] [-d BOUNDARY_DISTANCE] [-s STEP_SIZE] [-m LOSS_MULTIPLE]
                        data labels model

ADVERSARIAL PARAMETER ENCRYPTION This script encrypts the parameters of an input Pytorch model. The following dependencies must be installed: `json`, `torch`, `random`, `pickle`, `argparse`, `numpy`, and `datetime`.

positional arguments:
  data                  Filepath for encryption data (or secret key in decrypt mode)
  labels                Filepath for encryption labels (or location to save decrypted model in decrypt mode)
  model                 Filepath for torchscript model to encrypt/decrypt

options:
  -h, --help            show this help message and exit
  --disable-gpu         Disable GPU use
  --json-key            Save secret key as JSON
  --pt-data             Pytorch dataset files instead of numpy
  --decrypt-mode        Decrypt model with key
  --output-model OUTPUT_MODEL
                        Filepath to save model after encryption
  --output-key OUTPUT_KEY
                        Filepath to save secret key for decryption
  -l MAX_LAYERS, --max-layers MAX_LAYERS
                        Default 25. Maximum number of layers to encrypt
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Default 32. Number of examples to process at once
  -p MAX_PARAMS, --max-params MAX_PARAMS
                        Default 25. Maximum number of parameters to encrypt per layer
  -d BOUNDARY_DISTANCE, --boundary-distance BOUNDARY_DISTANCE
                        Default 0.1. Set from 0 to 0.5. 
                        0 = encrypted parameters can be anywhere in range of existing parameters. 
                        0.5 = encrypted parameters must be existing parameter range's midpoint.
  -s STEP_SIZE, --step-size STEP_SIZE
                        Default 0.1. Step size per gradient-based update
  -m LOSS_MULTIPLE, --loss-multiple LOSS_MULTIPLE
                        Default 5. Stops training when loss has grown by N times
```

Some notes explaining the above options: 
- The **positional arguments are different for encryption mode and decryption mode**. In encryption mode, the arguments in order are `data_source.npy label_source.npy model_source.pt`. In decryption mode, the arguments in order are `decryption_key.pkl output_model_filepath.pt model_to_decrypt.pt`.
- A key aim of the algorithm is to keep encrypted (modified) parameters indistinguishable from unencrypted (original) parameters. To do this, it keeps encrypted parameter values within certain boundaries set within the range of existing parameter values in each layer. These **boundaries are computed using the boundary distance ($\beta$) as follows**:
  - $B_{low} = \min{W_l} + \beta \cdot (\max{W_l} - \min{W_l})$ - where $B_{low}$ is the lowest acceptable encrypted parameter value and $W_l$ represents the parameters of the $lth$ layer. 
  -  $B_{high} = \max{W_l} - \beta \cdot (\max{W_l} - \min{W_l})$ - where $B_{high}$ is the highest acceptable encrypted parameter value.
- The loss multiple sets the algorithm to stop encryption early if the loss has been raised (performance has been deteriorated) sufficiently. Ex: If you set this value to 5, then parameters stop being modified (encrypted) when the average loss across batches is 5 times higher than the loss prior to encryption. 

</details>







<details><summary><h3>Module-based Usage</h3></summary>

You can also import the Python script in `adv_params/adv_params_pt.py` as a module in your own Python scripts. The following dependencies must be installed: `json`, `torch`, `random`, `pickle`, `argparse`, `numpy`, and `datetime`.

Useful objects to import are:
- `EncryptionUtils`: Class to encrypt model parameters. 
- `get_layer_set`: Randomly selects model parameters for encryption. 
- `decrypt_parameters`: Decrypts a model's parameters given a secret key. 
- `secret_formatter`: Saves the secret key in a JSON/pickle format.

Here is some example code **showing how models can be encrypted**. Put this file in the same directory as `adv_params_pt.py`
```python
import torch
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from adv_params_pt import EncryptionUtils, get_layer_set

# Get a torch.nn.Module or torch.jit.ScriptModule however you want
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

# Load your dataset however you want
dataset = TensorDataset(YOUR_DATA_HERE, YOUR_LABELS_HERE)
encrypt_data = DataLoader(dataset, batch_size=32)


# Declare hyperparameters
max_layers = 25                 # max layers to encrypt
max_params = 25                 # max parameters to encrypt per layer
step_size = 0.1                 # adjusts gradient update size
loss_multiple = 5               # stop encrypting when loss raised 5x
boundary_distance = 0.1         # set 0-0.5. See docs for details
device = torch.device('cpu')

# Set a max loss to stop at
x,y = next(iter(encrypt_data))
with torch.no_grad():
    loss = F.cross_entropy(model(x), y.long())
loss_threshold = loss.item() * loss_multiple


# Encrypt the model
encrypt_layers = get_layer_set(max_layers, model) 
instance = EncryptionUtils(model, encrypt_data, encrypt_layers,
                          max_params, loss_threshold, step_size, 
                          boundary_distance, device)
secret = instance.encrypt_parameters() 


# Save your secret (decryption key) and encrypted model parameters at the end.
torch.save(model, 'encrypted_model.pkl')
f = open('decrpytion_key.pkl', 'wb')
pickle.dump(secret, f)
f.close()
```

A **note about the boundary distance hyperparameter**:
- A key aim of the algorithm is to keep encrypted (modified) parameters indistinguishable from unencrypted (original) parameters. To do this, it keeps encrypted parameter values within certain boundaries set within the range of existing parameter values in each layer. These boundaries are computed using the boundary distance ($\beta$) as follows:
  - $B_{low} = \min{W_l} + \beta \cdot (\max{W_l} - \min{W_l})$ - where $B_{low}$ is the lowest acceptable encrypted parameter value and $W_l$ represents the parameters of the $lth$ layer. 
  -  $B_{high} = \max{W_l} - \beta \cdot (\max{W_l} - \min{W_l})$ - where $B_{high}$ is the highest acceptable encrypted parameter value.

Finally, this code snippet shows **how to decrypt models**:
```python
import torch
import pickle
from adv_params_pt import decrypt_parameters

# Load encrypted data
model = torch.load(model, 'encrypted_model.pkl')
secret = pickle.load('decryption_key.pkl')

# Decrypt parameters and save model
decrypt_parameters(model, secret)
torch.save(model, 'decrypted_model.pkl')
```

</details>

</details>











