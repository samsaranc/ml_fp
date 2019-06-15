A tar file of the entire split dataset is available [here](https://drive.google.com/file/d/1DnfVy4HbaExCISE8ePMT708DOpLRX1RC/view?usp=sharing).

To load the classifier, typically you need to read the file `model.pth`
which needs to be in the same directory as a `index2class.pth` file.

The directory `models` has all of the different versions of the classifier. Each sibdirectory contains the categories used (2 or 3 of Artist, Scientist, and Fashion), top accuracy score (out of 100%), and whether the ResNet model was pre-trained (P or NP).  
