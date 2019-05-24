To load the classifier, typically you need to read the file `model.pth`
which needs to be in the same directory as a `index2class.pth` file.

The directory `models` has all of the different versions of the
classifier (stages 1-3, with 1 and 3 having more than one version) . it doesn't matter which one you use; I'm editing `Test.py`to
load one of the `s3` versions.
