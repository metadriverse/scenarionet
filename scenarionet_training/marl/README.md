# ScenarioNet-MARL Temporary Readme

## Installation

```bash
conda create -n snmarl python=3.7 -y
conda activate snmarl


cd ~/metadrive
git pull -p
git checkout neurips-marl
pip install -e .


cd ~/scenarionet
pip install -e .


pip install ray[all]==2.2.0
pip install gym==0.19.0
pip install -U wandb

#conda install cudatoolkit=11.8 -c pytorch -y
#conda install cudatoolkit=11.8 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install torch torchvision torchaudio
pip uninstall typing_extensions
pip install typing_extensions
python -c "import torch;print(torch.cuda.is_available())"  # Check whether pytorch is installed correctly

#pip install numpy==1.19.5 -U
#pip uninstall PIL
#pip uninstall Pillow
#pip install Pillow
#pip install six -U
#pip uninstall requests
#pip install requests
#pip uninstall urllib3
#pip install urllib3
#conda install chardet -y

cd ~/scenarionet/scenarionet_training/marl
python 


# pip install -U tensorflow-probability==0.11.1

# CUDA 10.10
#conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
#pip install -U tensorflow==2.3.1

# CUDA 11.4
#conda install pytorch torchvision torchaudio cudatoolkit=11.3 cudnn -c pytorch
#pip install -U tensorflow==2.5.0

# CUDA 11.8 / 11.7
# ...


```
