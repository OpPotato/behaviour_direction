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

# Qwen3
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B

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

# Run with thinking evaluation (prefills <think> for chain-of-thought)
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --eval_thinking

# Disable thinking mode for Qwen3 (prefills <think></think> to skip chain-of-thought)
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --disable_thinking

## Extended Generation (Agent Role/Scenario/Choice format)
# Generate responses for extended scenarios with steering (no scoring - local only)
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --skip_generate --skip_select --skip_eval_ab --skip_eval_open --generate_extended

# With custom dataset path
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --skip_generate --skip_select --skip_eval_ab --skip_eval_open --generate_extended --extended_dataset datasets/raw/survival-instinct/dataset_extended.json

# Limit number of scenarios (useful for testing)
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --skip_generate --skip_select --skip_eval_ab --skip_eval_open --generate_extended --extended_limit 5

# Can be run locally
python visualize_results.py --run_dir pipeline/runs/Llama-2-7b-chat-hf/survival-instinct

python -m run_completion --model_path Qwen/Qwen3-8B --enable_tools --max_tool_calls 1

# Disable thinking mode for Qwen3 (prefills <think></think> to skip chain-of-thought)
python -m run_completion --model_path Qwen/Qwen3-8B --disable_thinking


# If running out of space, can run this for previous models
rm -rf /workspace/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf

rm -rf /root/.cache/huggingface/hub/models--meta-llama--*
rm -rf /root/.cache/huggingface/hub/models--google--*