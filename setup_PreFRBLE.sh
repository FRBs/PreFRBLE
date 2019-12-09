#!/bin/bash

DIR=$PWD/PreFRBLE_venv
PACKAGE=$PWD/PreFRBLE

mkdir $DIR
virtualenv -p python3 $DIR
source $DIR/bin/activate

pip install ipython, jupyter

cp -r $PACKAGE $DIR
pip install $DIR/PreFRBLE_package
