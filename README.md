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

## Available variants
# Llama 2 variants
python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf --scoring_model anthropic/claude-opus-4.5
python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-13b-chat-hf

# Llama 3
python -m pipeline.run_pipeline --model_path meta-llama/Meta-Llama-3-8B-Instruct

# Gemma
python -m pipeline.run_pipeline --model_path google/gemma-7b-it

# Qwen
python -m pipeline.run_pipeline --model_path Qwen/Qwen-7B-Chat

# Results saved to pipeline/runs/Llama-2-7b-chat-hf/survival-instinct/

## Evaluation options
# Run only evaluations (skip generation/selection)
python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf --skip_generate --skip_select

# Run only open-ended evaluation with GPT scoring
python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf --skip_generate --skip_select --skip_eval_ab

# Run only A/B evaluation
python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf --skip_generate --skip_select --skip_eval_open

# Show top-k predicted tokens in A/B evaluation (useful for debugging)
python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf --show_top_logits --top_k 10

# Can be run locally
python visualize_results.py --run_dir pipeline/runs/Llama-2-7b-chat-hf/survival-instinct

# If running out of space, can run this for previous models
rm -rf /workspace/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf