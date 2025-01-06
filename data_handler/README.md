# Sleep stage classification data handler


This package is used to handle the data for the project. 
It is responsible for downloading the data, cleaning it, preprocessing it, and saving it in a format that can be used by the models.
Downloading the data requires the command line tools wget and rar.
The package is automatically called by the model scripts if the data is not in the specified folder.


## Dependencies
- `wget` for downloading files from URLs
- `unrar` for extracting `.rar` files

Ensure these are installed on your system before running the package.