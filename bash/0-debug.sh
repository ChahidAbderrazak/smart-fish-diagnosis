#/bin/bash
# pip install flake8 black
#  debuggin the code
echo && echo "[${PROJECT_NAME}][dev] Linting the codes."
#------------------------------------------------------------------------
echo "Debuging the code..."
python -m flake8 ./src --count --select=E9,F63,F7,F82 --ignore=F541,W503 --show-source --statistics
python -m flake8 ./src --count --ignore=F541,W503 --max-complexity=10 --max-line-length=127 --statistics
python -m black --line-length 79 . 