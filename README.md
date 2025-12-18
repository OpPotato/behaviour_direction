# behaviour_direction

Workflow
# Clone/copy your repo
cd /workspace/behaviour_direction

# Setup (creates venv + installs deps)
./setup.sh

# Activate venv
source .venv/bin/activate

# Run pipeline
python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf

# Results saved to pipeline/runs/Llama-2-7b-chat-hf/survival-instinct/


# Can be run locally
python visualize_results.py --run_dir pipeline/runs/Llama-2-7b-chat-hf/survival-instinct