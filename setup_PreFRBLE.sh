#!/bin/bash

DIR=$PWD/PreFRBLE_venv
PACKAGE=$PWD/PreFRBLE

rm -rf $DIR
mkdir $DIR
virtualenv -p python3 $DIR
source $DIR/bin/activate

pip install ipython jupyter scipy

cp -r $PACKAGE $DIR
pip install $DIR/PreFRBLE
