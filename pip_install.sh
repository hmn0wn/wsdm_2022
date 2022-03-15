#!/bin/sh
#apt-get update  # To get the latest package lists
#etc.

#sudo pip install virtualenv
#sudo apt install python3-venv
#python3 -m venv bsa
#sudo python3 -m venv env_bsa
#deactivate 
#source env_bsa/bin/activate
apt-get install -qq gcc-5 g++-5 -y


sudo pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 #-f https://download.pytorch.org/whl/torch_stable.html
sudo pip install torch-scatter #-f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
sudo pip install torch-sparse #-f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
sudo pip install torch-cluster #-f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
sudo pip install torch-spline-conv #-f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
#sudo pip install torch-geometric
sudo pip install memory_profiler
#!apt-get install python-numpy python-scipy
sudo pip install cython==0.29.21 eigency==1.77 numpy==1.18.1 torch-geometric==1.6.3 tqdm==4.56.0 ogb==1.2.4