# Baselines in tensorflow-keras

This project collects the four baselines which were originally written in tensorflow-keras.
These are:
1. GraphSleepNet
2. MSTGCN
3. JK-STGCN
4. cVAN

The code for all four projects is publicly available.
We decided to keep the original projects in tensorflow-keras, in order to avoid porting errors.
While we tried to change the original code as little as possible, there are essentially seven changes with respect to the original projects:
1. We combined the four tf-keras projects due to their similar structure, forming one project with shared dependencies.
2. The data handling functionality was moved to a separate package ("data_handler"), because the functionality is shared with the models implemented in pytorch.
3. The original projects use tensorflow versions which are no longer available. We therefore updated some functions to work with a newer tensorflow version (2.12). 
4. We removed the issue of data leakage present in all four original projects. In these, the testing set was monitored during training for early stopping.
5. We changed how data is read in, as the original read-in functionality was not sufficient for larger datasets.
6. We built a wrapper around the original model training to facilitate training multiple repeats on multiple datasets.
7. We changed how results are stored. In particular, we save confusion matrices for every trained model to facilitate the model comparison framework in our study.
