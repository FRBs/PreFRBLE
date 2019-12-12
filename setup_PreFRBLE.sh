#!/bin/bash


## directiories for virtual environment and where source code is found
DIR=$PWD/venv_PreFRBLE
PACKAGE=$PWD/PreFRBLE


## create directory of virtual environment (delete old version)
rm -rf $DIR
mkdir $DIR
virtualenv -p python3 $DIR
source $DIR/bin/activate

## install python packages used not by PreFRBLE, but by notebooks
pip install ipython jupyter scipy  

## copy source code to virtual environemnt and install
cp -r $PACKAGE $DIR
pip install $DIR/PreFRBLE

## PreFRBLE can now be used in python within the virtual environment
## activate environment by changing into parent folder and execute
##
##    source .activate_PreFRBLE
##
## in python, you can now use PreFRBLE functions by calling
##
##    import PreFRBLE
##
