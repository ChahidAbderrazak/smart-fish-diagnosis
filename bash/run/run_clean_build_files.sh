#!/bin/bash
clear
echo && echo " #################################################" 
echo " ##          OBJECT DETECTION PROJECT           " 
echo " ## Clean the build files "
echo " #################################################" && echo 

#--------------------------------------------------------
echo && echo " -> Clean the __pycache__ folders "
sudo rm -rfv `find -type d -name *__pycache__*`

echo && echo " -> Clean the .pyc files "
sudo rm -fv `find -type f -name *.pyc`

echo && echo " -> Clean the checkpoint folders "
sudo rm -rfv `find -type d -name *checkpoint*`

echo && echo " -> Clean the pytest_cache folders "
sudo rm -rfv `find -type d -name *.pytest_cache*`


echo && echo " -> Done"