# MSA-CNN

This repository contains the code for the paper "MSA-CNN: A Lightweight Multi-Scale CNN with Attention for Sleep Stage Classification".
It comprises code for the MSA-CNN model in pytorch, as well as baselines in pytorch and tensorflow-keras. 
The pytorch models are located in the `models-pytorch` directory, while the tensorflow-keras models are located in the `models-tf-keras` directory.
These directories function as subprojects, each with its own set of required packages.
Both directories make use of the local `data_handler` package, which contains the data handling functionality shared by all models.
The `data_handler` package is used to download, prepare, and preprocess the data.

