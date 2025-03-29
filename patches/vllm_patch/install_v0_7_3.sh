# this script is used to install vllm 0.7.3 with a customized requirement version that meet sglang required

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd $SCRIPT_DIR/../../..

git clone https://github.com/vllm-project/vllm.git
pip uninstall -y vllm

# align with sglang
pip install transformers==4.48.3

cd vllm
git checkout v0.7.3
cp $SCRIPT_DIR/requirements-common.txt .
export CCACHE_DIR=/scratch/ccahe
python setup.py develop
cd ..
