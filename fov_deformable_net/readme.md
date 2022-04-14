
## This is the official Pytorch implementation of FoV deformable network.

### Prerequisite

* Python 3.7
* Matlab
* Other python packages are downloaded as follows:
```python
pip install -r requirements.txt
```
*None*: Please make sure your machine has a GPU, and its driver version is comply with the CUDA version! This will reduce the problems when installing the DCNv2 module later.

* #### The Deformable ConvNets V2 (DCNv2) module in our code adopts [Xujiarui's Implementation](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0). We recommand you recompile the code according to your machine and python environment as follows:

```python
cd ~/dcn
python setup.py develop
```

This may cause many issues, please open Issues and ask me if you have any problems!

### 1. prepare the dataset of your camera by:

```python
python dataset_generator.py
```

Note that the path information in this file needs update to the path of your computer:

```python
date_ind = "2022xxxx" # date information for h5py file
dataset_type = "valid" # type of dataset "train" or "valid"
camera_idx = "camera0x" # index of camera "camera01" to "camera05" 
base_path = "./synthetic_datasets" # system path 
input_dir = "input_rgb_2022xxxx" # input data dir
label_dir = "label_rgb" # label data dir
if_mask = False # whether add mask
```

### 2. Check the option file information

* #### Checking the data path and other hyper-parameters for training   

Note: The training information and the test information are in the same option.py file!

### 3. Training the FoV deformable network

```python
python train.py
```

### 4. Test on the actual photographs of your camera

```python
python test_real.py
```

