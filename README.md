[![DOI:10.3390/s22239547](https://img.shields.io/badge/DOI-10.3390/s22239547-0064A4.svg)](https://doi.org/10.3390/s22239547)
[![WEB:mbenomar.com](https://img.shields.io/badge/WEB-mbenomar.com-FFD200.svg)](https://mbenomar.com)
[![WEB:hero.eng.uci.edu](https://img.shields.io/badge/WEB-HERO%20Lab-48159A.svg)](https://hero.eng.uci.edu)

# EEG Biometrics for Raspberry Pi
Library for EEG Biometrics project for HERO Lab - UC Irvine

## Git clone repository with SSH Key
Generate key in Raspberry Pi
```
ssh-keygen -t ed25519 -C "mohamed_3151@hotmail.com"
```

Copy SHH Key and add it to Github Account
```
sudo nano ~/.ssh/id_ed25519.pub
```

Clone repository to Raspberry Pi
```
git clone git@github.com:MohamedBenomar/EEG_Biometrics_Raspberry_Pi.git
cd EEG_Biometrics_Raspberry_Pi
```


## Install python 3.7 in Raspberry Pi

Install pyenv
```
curl https://pyenv.run | bash
```

Add pyenv to bash
```
sudo nano ~/.bashrc
```
```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
```

Restart the terminal
```
exec $SHELL
```

Install dependencies
```
sudo apt-get install --yes libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libgdbm-dev lzma lzma-dev tcl-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev wget curl make build-essential openssl
```

Restart the terminal
```
exec $SHELL
```

Update pyenv
```
pyenv update
```

Install python versions
```
pyenv install --list
pyenv install 3.7.12
```

Set python verion:
`pyenv local 3.7.12` or `pyenv shell 3.7.12` or `pyenv global 3.7.12`

To uninstall (in case you followed along and changed your mind)
```
rm -fr ~/.pyenv
```
Remove lines from .bashrc (sudo nano ~/.bashrc)


## Requirements

- `Python` == 3.7 or 3.8
- `tensorflow` == 2.X (verified working with 2.0 - 2.3, both for CPU and GPU)
- `keras` >= 2.8.0

- `mne` >= 0.24.1
- `librosa` >= 0.9.1
- `sklearn` >= 1.0.2
- `scikit-learn` >= 0.20.1
- `pandas` >=  1.4.1
- `numpy` >= 1.22.0

```
pip install sklearn
pip install scikit-learn
pip install pandas
pip install numpy
pip install keras
pip install --upgrade adafruit-python-shell
wget https://raw.githubusercontent.com/adafruit/Raspberry-Pi-Installer-Scripts/master/raspi-blinka.py
sudo python raspi-blinka.py
```


## Install tensorflow in Raspberry Pi

Make a virtual environment:
```
python3 -m pip install virtualenv
python3 -m virtualenv env
source env/bin/activate
```

Run the commands from https://github.com/PINTO0309/Tensorflow-bin/#usage:
```
sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran libgfortran5 libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev liblapack-dev cython3 libatlas-base-dev openmpi-bin libopenmpi-dev python3-dev
pip install -U wheel mock six
```

Select the .whl from https://github.com/PINTO0309/Tensorflow-bin/tree/main/previous_versions
```
wget https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/main/previous_versions/download_tensorflow-2.5.0rc0-cp37-none-linux_armv7l.sh
sudo chmod +x download_tensorflow-2.5.0rc0-cp37-none-linux_armv7l.sh
./download_tensorflow-2.5.0rc0-cp37-none-linux_armv7l.sh
sudo pip uninstall tensorflow
pip uninstall tensorflow
pip install tensorflow-2.5.0rc0-cp37-none-linux_armv7l.whl
```

Restart the terminal
```
exec $SHELL
```

Reactivate virtual environment
```
source env/bin/activate
```

Uninstall hdf5 and reinstall it
```
pip uninstall h5py
HDF5_VERSION=1.10.6 pip install --no-binary=h5py h5py==3.1.0
```


## Install MNE in Raspberry Pi

```
pip install mne
```

## Install Adafruit Packages

### Adafruit_CircuitPython_MCP3008

CircuitPython library for the MCP3xxx series of analog-to-digital converters.

```
sudo pip3 install adafruit-circuitpython-mcp3xxx
```

### Adafruit_CircuitPython_MCP4725

CircuitPython module for the MCP4725 digital to analog converter.

```
sudo pip3 install adafruit-circuitpython-mcp4725
```

## Code Structure

### ppeeg

Python module with some EEG preprocessing functions for biometrics applications

This module requires some libraries:
- `pyprep` (Preprocessing Pipeline): Standardized preprocessing for large-scale EEG analysis. DOI: 10.3389/fninf.2015.00016
- `MNE`: Open-source Python package for exploring, visualizing, and analyzing human neurophysiological data: MEG, EEG, sEEG, ECoG, NIRS, and more.
- `Librosa`: Python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

### ClassifiersModelsEEG

Python module with different Deep Learning Classifiers models for EEG biometrics

This module requires `keras` and `tensorflow`

### ClassifierEEG

Python module that makes the classification for the EEG samples read from the MCP3008

### utilsEEG

Python module with some usefull tools for the classification models

### main

Main script of the EEG Biometrics device. Use the OpenBCI headset to collect EEG signals from a subject and by the way of different tools and Deep Learning models, predict the person identity.
