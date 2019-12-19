#!/bin/bash

## exit if any command fails
set -e

## path of thi bash script, should be placed in top folder of PreFRBLE
BASE_DIR=$(dirname $0)

## directiories for virtual environment and where source code is found
DIR=$BASE_DIR/venv_PreFRBLE
PACKAGE=$BASE_DIR/PreFRBLE

read -p "Do you want to create a new virtual environment? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    ## create directory of virtual environment (delete old version)
    rm -rf $DIR
    mkdir $DIR
    virtualenv -p python3 $DIR

    source $DIR/bin/activate

    ## install python packages used not by PreFRBLE, but by notebooks. If this fails, continue with install_PreFRBLE.sh
    pip3 install numpy cython ipython jupyter scipy pandas statsmodels    
fi

## install PreFRBLE
bash $BASE_DIR/install_PreFRBLE.sh


## PreFRBLE can now be used in python within the virtual environment
## activate environment by changing into parent folder and execute
##
##    source .activate_PreFRBLE
##
## in python, you can now use PreFRBLE functions by calling
##
##    import PreFRBLE
##
