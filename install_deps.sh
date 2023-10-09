#!/bin/bash

apt-get update
apt-get install -y --no-install-recommends build-essential libarchive-dev libboost-dev python3-dev python3-pip rapidjson-dev software-properties-common
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
mkdir /root/.conda
bash Anaconda3-2021.11-Linux-x86_64.sh -b -p /opt/conda
/opt/conda/bin/conda install cmake

export PATH=/opt/conda/bin/:$PATH
cmake --version
