# Code for AI4S_CG2309 
## Authors, institution, location
Below is the information for the authors.
+ *Author* Ji Li and Chao Wang
+ *Institution_1* Academy for Multidisciplinary Studies, Capital Normal University, Beijing
+ *Institution_2* Kansas Medical Center, USA
-------
## Brief description of your algorithm and a mention of the competition
The repo includes all the codes for the competition track *CG2309*, which aims to develop the deep learning solver for high-dimensional Poisson equation and the operator learning solver for two-dimensional Poisson equation. 

For the first task, we follow the physics-informed neural networks (PINNs) framework to address the 10-dimenisonal Poisson equation. To mitigate the issue of high dimensionality and the high-frequency solution, we propose several modifications to the network architecture, including the separable subnetwork with sine activation function and the R3 (Retain-Resample-Release) sampling method for collection points.

For the second operator learning task, we propose the method

This code repository is uploaded for competition of CG2309 of AI4S.

## Installation instructions, including any requirements
See the ```requirement.txt``` to install the dependent packages and libraries.

+ Clone the github repository
```python
git clone https://github.com/Chilie/AI4S_CG2309.git
cd AI4S_CG2309
```
+ Use ```conda``` to construct the virtual environment
```python
conda create -n ai4s python=3.7
conda activate ai4s # activate the environment
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html # install cuda verison of pytorch
pip3 install -r requirements.txt # install the dependency 
# deactivate
```
+ In our code, we put the pretrianed models in the folder `checkpoints`

## Usage instructions
+ PINNs for Poisson equation
+ 10-d Poisson equation
+ Training code, for arguments to the training code, see details of the *.py file
```python
python3 main_with_time_d10.py
```
+ Test code, the checkpoint is stored in './checkpoints/DIM10/Seed_1234_layers_4_dim_hidden_32_dim_out_5.pth'
```python
python3 test_10d.py
```
+ 2-d Poisson equation
+ Training code, for arguments to the training code, see details of the *.py file
```python
python3 main_with_time_d2.py
```
+ Test code, the checkpoint is stored in './checkpoints/DIM2/Seed_1234_layers_3_dim_hidden_32_dim_out_3.pth'
```python
python3 test_2d.py
```

## Operator learning code

For this code, we require the following pytorch version, please follow the code to install the environment
```python
cd GIT # enter the folder
conda create -n git python=3.7
conda activate git # activate the environment
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt # install the dependency 
```

# The training data

The training data is hosted in baidunetdisk, you can download these files at https://pan.baidu.com/s/1tDOZw2uylySBs3fKbR4Ofw?pwd=ejnp 
Note that the folder structure should be hold and put it in the GIT folder.
+ Training code, run the following 
```python
python3 GIT_poisson.py
```
+ Test code, run the following
```python
python3 eval_poisson_GIT.py
```