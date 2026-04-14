set -e # exit script on error

source .venv/bin/activate

pip install -r requirements.txt

source ~/akantu/build/akantu_environement.sh

set +e # return to default shell behaviour 
