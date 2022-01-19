#!/bin/bash
set -eux

# Install nvidia drivers
# https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html
sudo yum groupinstall -y "Development Tools"
sudo yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
BASE_URL=https://us.download.nvidia.com/tesla
DRIVER_VERSION=470.82.01
curl -fSsl -O $BASE_URL/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
sudo sh NVIDIA-Linux-x86_64-$DRIVER_VERSION.run --silent

# Install the container toolkit
# Instructions from https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo

sudo yum clean expire-cache
sudo yum install -y nvidia-docker2
sudo systemctl restart docker

# Automatically start it on boot
sudo systemctl --now enable docker

# Sanity check
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
