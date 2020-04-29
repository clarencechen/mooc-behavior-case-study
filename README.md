# MOOC Student Course Trajectory Modeling

This is the Python 3 source code for of the MOOC Course Trajectory Modeling paper:

Chen, C., Pardos, Z. (2020) Applying Recent Innovations from NLP to MOOC Student Course Trajectory Modeling.

### Dependencies
- keras
- tensorflow
- pandas
- numpy
- h5py
- keras-transfomer

The above dependencies may be installed from the requirements file by running the following command:
```
pip3 install -r requirements.txt
```
To install the dependency keras-transformer you need to populate the submodule
```
git submodule update --init
```
then switch to the populated directory and run pip
```
cd keras-transformer
pip install .
```
Please note that the project requires Python >= 3.6.

### Dataset:

***Unfortunately, the MOOC dataset for this paper is not yet publicly available. Please contact Zachary Pardos for access to dataset.
