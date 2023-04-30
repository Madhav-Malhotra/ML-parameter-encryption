# ML Parameter Encryption

This repository has implementations of recent encryption algorithms for AI models. These algorithms aim to **prevent unauthorised users from accessing AI models.** My intention in creating these implementations is to make theoretical algorithms more production-ready and useful. 

The algorithms implemented are described in the following papers: 
1. A. Pyone, M. Maung, and H. Kiya, ‘Training DNN Model with Secret Key for Model Protection’, in 2020 IEEE 9th Global Conference on Consumer Electronics (GCCE), Kobe, Japan: IEEE, Oct. 2020, pp. 818–821. doi: [10.1109/GCCE50665.2020.9291813](https://doi.org/10.1109/GCCE50665.2020.9291813).
2. M. Alam, S. Saha, D. Mukhopadhyay, and S. Kundu, ‘Deep-Lock: Secure Authorization for Deep Neural Networks’. arXiv, Aug. 13, 2020. Accessed: Apr. 29, 2023. \[Online\]. Available: http://arxiv.org/abs/2008.05966
3. M. Xue, Z. Wu, J. Wang, Y. Zhang, and W. Liu, ‘AdvParams: An Active DNN Intellectual Property Protection Technique via Adversarial Perturbation Based Parameter Encryption’. arXiv, May 28, 2021. Accessed: Apr. 29, 2023. \[Online\]. Available: http://arxiv.org/abs/2105.13697

## AdvParams

This algorithm currently has a Pytorch implementation for any model derived from the [`torch.jit.ScriptModule`](https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule) class. Models stored using [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class can easily be converted to and from this format. 

### Script-based Usage

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
python3 adv_params.py encrypt_imgs.npy encrypt_labels.npy model_scripted.pt
```

As an output, the script will save a secret key necessary to decrypt the model in `decryption_key.pkl`. Also, it will save the encrypted model to `encrypted_model.pt`. You can **decrypt the model** with this data using this command. It specifies that the script should get the decryption key and encrypted model from the above files, decrypt the model, and then save the decrypted model as `decrypted_model.pt`. 
```bash
python3 adv_params.py decryption_key.pkl decrypted_model.pt encrypted_model.pt --decrypt-mode
```

To see other script options, run 
```
python3 adv_params.py --help

usage: adv_params_pt.py [-h] [--disable-gpu] [--json-key] [--pt-data] [--decrypt-mode] [--output-model OUTPUT_MODEL] [--output-key OUTPUT_KEY] [-l MAX_LAYERS] [-b BATCH_SIZE] [-p MAX_PARAMS]
                        [-d BOUNDARY_DISTANCE] [-s STEP_SIZE] [-m LOSS_MULTIPLE]
                        data labels model

ADVERSARIAL PARAMETER ENCRYPTION This script encrypts the parameters of an input Pytorch model. The following dependencies are required: `json`, `torch`, `random`, `pickle`, `argparse`, `numpy`,
and `datetime`. This file can also be imported as a module with these objects: * EncryptionUtils: Class to encrypt model parameters. * get_layer_set: Randomly selects model parameters for
encryption. * decrypt_parameters: Decrypts a model's parameters given a secret key. * secret_formatter: Saves the secret key in a JSON/pickle format.

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
                        Maximum number of layers to encrypt
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Number of dataset images to process at once
  -p MAX_PARAMS, --max-params MAX_PARAMS
                        Maximum number of parameters to encrypt per layer
  -d BOUNDARY_DISTANCE, --boundary-distance BOUNDARY_DISTANCE
                        Set from 0 to 0.5. 0 = extreme where encrypted parameter values can be anywhere in range of existing parameter values. 0.5 = extreme where encrypted parameter values can
                        only be at the midpoint of the range of existing parameter values.
  -s STEP_SIZE, --step-size STEP_SIZE
                        Step size per gradient-based update
  -m LOSS_MULTIPLE, --loss-multiple LOSS_MULTIPLE
                        Stops training when the loss has grown by N times
```