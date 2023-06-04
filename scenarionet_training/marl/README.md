

## Installation

```bash
conda create -n copo python=3.7
conda activate copo

cd metadrive
git checkout main
git pull -p
pip install -e .

cd drivingforce
git checkout copo
git pull -p
pip install -e .

pip install ray[all]==1.11.0
pip install gym==0.19.0
pip install -U wandb
pip install numpy==1.19.5

# pip install -U tensorflow-probability==0.11.1

# CUDA 10.1
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install -U tensorflow==2.3.1

# CUDA 11.4
conda install pytorch torchvision torchaudio cudatoolkit=11.3 cudnn -c pytorch
pip install -U tensorflow==2.5.0
```
