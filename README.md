GW machine learning final project.

Data sorted into Artist, Scientist, and Fashion categories. A tar file of the entire split dataset used for training is available [here](https://drive.google.com/file/d/1DnfVy4HbaExCISE8ePMT708DOpLRX1RC/view?usp=sharing).

To load the classifier, typically you need to read the file `model.pth`
which needs to be in the same directory as a `index2class.pth` file.

The directory `best_models` has all of the best versions of the classifiers. Each subdirectory contains the categories used (2 or 3 of Artist, Scientist, and Fashion), top accuracy score (out of 100%), and whether the ResNet model was pre-trained (P or NP).  
