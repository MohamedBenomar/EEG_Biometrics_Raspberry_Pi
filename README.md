# EEG Biometrics for Raspberry Pi
Library for EEG Biometrics project for HERO Lab - UC Irvine


# Requirements

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
sudo -H pip3 install sklearn
sudo -H pip3 install scikit-learn
sudo -H pip3 install pandas
sudo -H pip3 install numpy
sudo -H pip3 install --upgrade adafruit-python-shell
wget https://raw.githubusercontent.com/adafruit/Raspberry-Pi-Installer-Scripts/master/raspi-blinka.py
sudo python3 raspi-blinka.py
```

## Install tensorflow in Raspberry Pi

```
# get a fresh start
sudo apt-get update
sudo apt-get upgrade
# remove old versions, if not placed in a virtual environment (let pip search for them)
sudo pip uninstall tensorflow
sudo pip3 uninstall tensorflow
# install the dependencies (if not already onboard)
sudo apt-get install gfortran
sudo apt-get install libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get install libatlas-base-dev libopenblas-dev libblas-dev
sudo apt-get install openmpi-bin libopenmpi-dev
sudo apt-get install liblapack-dev cython
sudo pip3 install keras_applications==1.0.8 --no-deps
sudo pip3 install keras_preprocessing==1.1.0 --no-deps
sudo pip3 install -U --user six wheel mock
sudo -H pip3 install pybind11
sudo -H pip3 install h5py==2.10.0
# upgrade setuptools 40.8.0 -> 52.0.0
sudo -H pip3 install --upgrade setuptools
# download the wheel
wget https://github.com/Qengineering/Tensorflow-Raspberry-Pi/raw/master/tensorflow-2.1.0-cp37-cp37m-linux_armv7l.whl
# install TensorFlow
sudo -H pip3 install tensorflow-2.1.0-cp37-cp37m-linux_armv7l.whl
# and complete the installation by rebooting
sudo reboot
```

## Install MNE in Raspberry Pi

```
pip install mne
```

# Packages

## ppeeg

Python module with some EEG preprocessing functions for biometrics applications

This module requires some libraries:
- `pyprep` (Preprocessing Pipeline): Standardized preprocessing for large-scale EEG analysis. DOI: 10.3389/fninf.2015.00016
- `MNE`: Open-source Python package for exploring, visualizing, and analyzing human neurophysiological data: MEG, EEG, sEEG, ECoG, NIRS, and more.
- `Librosa`: Python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

## Adafruit_CircuitPython_MCP3008


CircuitPython library for the MCP3xxx series of analog-to-digital converters.

```
sudo pip3 install adafruit-circuitpython-mcp3xxx
```

## Adafruit_CircuitPython_MCP4725

CircuitPython module for the MCP4725 digital to analog converter.

```
sudo pip3 install adafruit-circuitpython-mcp4725
```

## ClassifiersModelsEEG

Python module with different Deep Learning Classifiers models for EEG biometrics

This module requires `keras` and `tensorflow`

## ClassifierEEG

Python module that makes the classification for the EEG samples read from the MCP3008

## utilsEEG

Python module with some usefull tools for the classification models

# main

Main script of the EEG Biometrics device. Use the OpenBCI headset to collect EEG signals from a subject and by the way of different tools and Deep Learning models, predict the person identity.
