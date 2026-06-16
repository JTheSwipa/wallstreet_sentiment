# Config paths
HF_MODEL_DIR="/leonardo_scratch/large/userinternal/$USER/public/models/hub"

# Create python3 env
cd /leonardo_scratch/fast/tra26_bbs
module load python/3.11.6--gcc--8.5.0
python3 -m venv venv
module purge

# Install the required libraries
source /leonardo_scratch/fast/tra26_bbs/venv/bin/activate
pip3 install -r /leonardo_work/tra26_bbs/mceccar2/hpc_bbs_26/team_project/llm/requirements.txt

# Download HF models
mkdir -p $HF_MODEL_DIR
export HF_HOME=$HF_MODEL_DIR
export HF_HUB_CACHE=$HF_MODEL_DIR

huggingface-cli download mistralai/Mistral-Small-3.2-24B-Instruct-2506
huggingface-cli download BAAI/bge-m3
