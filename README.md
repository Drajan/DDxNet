# DDxNet: A Multi-Specialty Diagnostic Model for Clinical Time-Series
<!-- **DDxNet: A Multi-Specialty Diagnostic Model for Clinical Time-Series** -->

DDxNet is a novel fully convolutional neural network architecture for time-varying clinical data. We demonstrate it's effectiveness for a variety of diagnostic tasks involving different modalities (ECG/EEG/EHR), required level of characterization (abnormality detection/phenotyping) and data fidelity (single-lead ECG/22-channel EEG). Using multiple challenging benchmark problems with EEG, ECG and EHR, we show that DDxNet produces high-fidelity predictive models in all cases, and more importantly provides significant performance gains over methods specifically designed for each of those problems. The architecture is depicted in the figure below:
<!-- ## DDxNet Architecture and Performance -->
![DDxNet Architecture](https://github.com/Drajan/DDxNet/blob/master/figures/arch.png)

<!--**Results on Arrhythmia Classification:**
![Loss](https://github.com/Drajan/DDxNet/blob/master/figures/mit_loss.png)
![Accuracy](https://github.com/Drajan/DDxNet/blob/master/figures/mit_acc.png)

**Generalization Performance of Detecting Unseen Cardiac Diseases:**
![Accuracy](https://github.com/Drajan/DDxNet/blob/master/figures/incart.png)-->

-----
## Usage

*Example Usage:*
``python train.py``


## Benchmark Problems and Datasets

**EEG-based abnormality detection** - the TUH abnormal corpus can be downloaded from https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/
As part of pre-processing, MFCCs are extracted using the librosa library (https://librosa.github.io/librosa/)

**ECG-based arrhythmia classification** - the Physionet MIT-BIH arrhythmia database can be downloaded from https://physionet.org/physiobank/database/mitdb/

**ECG-based myocardial infarction detection** - the Physionet PTBDB database can be downloaded from https://physionet.org/physiobank/database/ptbdb/

For both ECG problems, the data is pre-processed as described in the paper 'ECG Heartbeat Classification: A Deep Transferable
Representation' (https://arxiv.org/pdf/1805.00794.pdf) and can be downloaded from kaggle (https://www.kaggle.com/shayanfazeli/heartbeat)

**EHR-based phenotyping of ICU patients** - the Physionet MIMIC-III EHR database can be downloaded from https://mimic.physionet.org/
The benchmark dataset for phenotyping was then prepared using the mimic3-benchmarks repository (https://github.com/YerevaNN/mimic3-benchmarks)


## Requirements
* python 3.6.8
* numpy = 1.16.2
* pandas = 0.24.1
* torch = 1.0.1
* scikit-learn = 0.20.2
* matplotlib = 3.0.2
* bokeh = 1.0.4
* h5py = 2.9.0
* pip = 19.0.3
* python-dateutil = 2.8.0
* sklearn = 0.0


## Citations

If you find DDxNet useful in your research, please cite the following paper:
```
@article{thiagarajan2020ddxnet,
  title={DDxNet: a deep learning model for automatic interpretation of electronic health records, electrocardiograms and electroencephalograms},
  author={Thiagarajan, Jayaraman J and Rajan, Deepta and Katoch, Sameeksha and Spanias, Andreas},
  journal={Scientific reports},
  volume={10},
  number={1},
  pages={1--11},
  year={2020},
  publisher={Nature Publishing Group}
}
```
