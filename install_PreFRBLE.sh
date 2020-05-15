#!/bin/bash


## path of this bash script, should be placed in top folder of PreFRBLE
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

## directiories for virtual environment and where source code is found
DIR=$BASE_DIR/venv_PreFRBLE
PACKAGE=$BASE_DIR/PreFRBLE

source $DIR/bin/activate

## copy source code to virtual environemnt 
cp -r $PACKAGE $DIR
## make necessary changes to filespace
sed -i  -e "s-/data/PreFRBLE-$BASE_DIR-g" $DIR/PreFRBLE/PreFRBLE/file_system.py  ## use - as separator to easily replace /
##  INSTALL
pip install $DIR/PreFRBLE



## PreFRBLE is now updated according to changes in $PACKAGE
## simply restart jupyter notebook to use new version


