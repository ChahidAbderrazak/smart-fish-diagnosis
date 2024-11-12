#!/bin/bash
clear
echo && echo " #################################################" 
echo " ##          OBJECT DETECTION PROJECT           " 
echo " ## Run code tests "
echo " #################################################" && echo 
pip install pytest-cov
clear

#--------------------------------------------------------
echo && echo " -> measure  tests coverage report"
pytest --cov=src/lib tests/ --verbose --durations=5 -vv


#### ----------------   NOTIFICATION MESSAGE -------------------------
# notify-send "Execution Finished!!"