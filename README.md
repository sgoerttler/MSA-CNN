# MSA-CNN

This repository contains the code for the following papers:

- **"MSA-CNN: A Lightweight Multi-Scale CNN with Attention for Sleep Stage Classification"**
- **"Retrieving Filter Spectra in CNN for Explainable Sleep Stage Classification"**

The project includes PyTorch models located in the `models-pytorch` directory as well as TensorFlow-Keras models located in the `models-tf-keras` directory.
The two directories function as subprojects, each with its own set of required packages.
Both directories make use of the local `data_handler` package, which contains the data handling functionality shared by all models.


## **Overview**  
The repository includes:  
- **Data Handling:** Code in package `data_handler` for downloading, preparing and preprocessing the data. The data will be stored in the `data` folder at the top level. 
- **MSA-CNN and Baselines in PyTorch:** Code in directory `models-pytorch` for running, training, and evaluating the models implemented using PyTorch. This directory also contains the code for retrieving the filter spectra in the Multi-Scale Module.
- **Baselines in TensorFlow Keras:** Code in directory `models-tf-keras` for running, training, and evaluating the baseline models implemented using Tensorflow Keras.

## **Installation & Dependencies**  
This repository contains two subprojects, one for PyTorch models and one for TensorFlow-Keras models.
The dependencies have to be installed separately for each subproject. 

### **Python Dependencies:**  
Install the necessary Python packages by navigating to either the `models-pytorch` or `models-tf-keras` directory and executing:
```bash
pip install -r requirements.txt
```

## **Usage**

### **PyTorch Models: Training and Testing**

The following PyTorch models are included in this project:
- MSA-CNN
- AttnSleep
- DeepSleepNet
- EEGNet

To run a PyTorch model, specify the model, the dataset, as well as model and training hyperparameters in a json configuration file located in the `models-pytorch/configs` directory.
The model can then be run by passing the configuration file as an argument to the main.py file:
```bash
python main.py configs/<model_name>.json
```
The following optional arguments can be passed to the main.py file:
- `--verbose`: verbose level, 0=silent, 1=progress bar, 2=one line per epoch
- `--gpu`: which gpu to use
- `--rerun_configs`: rerun configurations even if present in overview
- `--dry_run`: do not save results

If several models, datasets, or parameters are specified in the configuration file, the main.py file will run all combinations.
To suppress running all model-dataset combinations, each model can be paired with a dataset by adding the following parameter to the configuration file (requires the number of models and datasets to match):
```json
"model_data_pairing": "element_wise"
```
Configurations with short runtimes to test the setup are included in the configuration folder with the suffix `_TEST.json`.



The results will be stored in the `models-pytorch/results` directory. Specifically, the script will generate a configuration file-specific overview file in the `overview` folder containing the results of all model runs, as well as the more specific training details for each model run in the folder `epoch_details`.

### **TensorFlow-Keras Models: Training and Testing**

TensorFlow-Keras models, located in the `models-tf-keras` directory, can be run by executing:
```bash
python main_<model>.py "<model>_training"
```
where `<model>` is the name of the model to be run. The following models are available:
- `cVAN`
- `GraphSleepNet`
- `JK-STGCN`
- `MSTGCN`

Model hyperparameters can be adjusted in the respective `<model>.json` file located in the `models-tf-keras/configs` folder.
Similarly, the dataset configuration is specified by the `<model>_<dataset>.json`, while the training hyperparameters are specified in the `<model>_training.json` file.
Configurations with short runtimes to test the setup are specified in the `<model>_training_TEST.json` file.

### **Model Comparison**
The model comparison section has not been added yet. It will include a detailed comparison of the performance of different models on relevant benchmarks and datasets.

### **Filter Spectra Retrieval**
This project is based on the paper **"Retrieving Filter Spectra in CNN for Explainable Sleep Stage Classification"**. 
The code for retrieving the filter spectra in the Multi-Scale Module is located in the `models-pytorch` directory. 
The filter spectra can be retrieved by running:
```bash
python main_fourier_filters.py
```
This will run the multivariate and univariate models on the two datasets ISRUC-S3 and Sleep-EDF-20, retrieve the filter spectra for the MSA-CNN Multi-Scale Module, and analyze and visualize the results.




## **Datasets**

Downloaded and preprocessed datasets used in this study are stored in the `data` directory at the top level.
The following datasets are used in this project:
- ISRUC-S3
- Sleep-EDF-20
- Sleep-EDF-78


## **Citation**
If you use the code of the MSA-CNN model or the evaluation framework, please cite:
- Goerttler, S., Wang, Y., Eldele, E., Wu, M., & He, F. (2025). **MSA-CNN: A Lightweight Multi-Scale CNN with Attention for Sleep Stage Classification**. *arXiv preprint arXiv:2501.02949*.

Alternatively, if you use the code for the filter spectra retrieval, please cite:
- Goerttler, S., Wang, Y., Eldele, E., He, F., & Wu, M. (2025). **Retrieving Filter Spectra in CNN for Explainable Sleep Stage Classification**. *arXiv preprint arXiv:2502.06478*.