# behaviour_direction

## Quick Start

# Setup (creates venv + installs deps)
cd /workspace/behaviour_direction
./setup.sh
source .venv/bin/activate

# Run basic pipeline
python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf

## Supported Models

# Llama 2
python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf
python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-13b-chat-hf

# Llama 3
python -m pipeline.run_pipeline --model_path meta-llama/Meta-Llama-3-8B-Instruct

# Gemma
python -m pipeline.run_pipeline --model_path google/gemma-7b-it

# Qwen
python -m pipeline.run_pipeline --model_path Qwen/Qwen-7B-Chat
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B

# Results saved to pipeline/runs/Llama-2-7b-chat-hf/survival-instinct/

## Pipeline Control
### Skip Steps

# Run only evaluations (skip generation/selection)
python -m pipeline.run_pipeline --model_path <model> --skip_generate --skip_select

# Skip specific evaluation types
--skip_eval_ab      # Skip A/B evaluation
--skip_eval_open    # Skip open-ended evaluation

# Run only open-ended evaluation with GPT scoring
python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf --skip_generate --skip_select --skip_eval_ab

# Run only A/B evaluation
python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf --skip_generate --skip_select --skip_eval_open
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --direction_method reasoning --skip_generate --skip_select --skip_eval_open --generate_extended

### Debugging
# Show top-k predicted tokens during evaluation
python -m pipeline.run_pipeline --model_path <model> --show_top_logits --top_k 10

## Direction Methods
### A/B Pairs (Default)
Extracts directions from contrasting A/B response pairs.

### Reasoning-Based
Extracts directions from `<think>...</think>` reasoning traces. Requires dataset with `positive_reasoning` and `negative_reasoning` fields.

# Use last token before </think>
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --direction_method reasoning

# Use mean pooling over all reasoning tokens
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --direction_method reasoning --pooling_strategy mean

## Extended Generation

Generate responses for extended scenarios (Agent Role/Scenario/Choice format):

# Generate with all multipliers
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B \
  --skip_generate --skip_select --skip_eval_ab --skip_eval_open \
  --generate_extended

# Test with limited scenarios
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B \
  --direction_method reasoning --skip_generate \
  --generate_extended --extended_limit 5 --parse_extended

# Run with thinking evaluation (prefills <think> for chain-of-thought)
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --eval_thinking

## Extended Generation (Agent Role/Scenario/Choice format)
# Generate responses for extended scenarios with steering (no scoring - local only)
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --skip_generate --skip_select --skip_eval_ab --skip_eval_open --generate_extended

# With custom dataset path
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --skip_generate --skip_select --skip_eval_ab --skip_eval_open --generate_extended --extended_dataset datasets/raw/survival-instinct/dataset_extended.json

# Limit number of scenarios (useful for testing)
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --skip_generate --skip_select --skip_eval_ab --skip_eval_open --generate_extended --extended_limit 5 --parse_extended

## Reasoning-Based Direction Generation
# Alternative to A/B pairs: extract directions from <think>...</think> reasoning traces
# Requires dataset_extended.json to have "positive_reasoning" and "negative_reasoning" fields

# Generate direction using reasoning traces (last token before </think>)
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --direction_method reasoning --show_top_logits --top_k 10 --generate_extended --extended_limit 5

# With mean pooling over all reasoning tokens
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --direction_method reasoning --pooling_strategy mean --show_top_logits --top_k 10 --generate_extended --extended_limit 5

# Skip generation and use existing reasoning direction
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --direction_method reasoning --skip_generate --skip_select --generate_extended

# Results saved to:
#   - Direction: pipeline/runs/<model>/<behavior>/direction_reasoning.pt
#   - Artifacts: pipeline/runs/<model>/<behavior>/generate_directions_reasoning/
#   - Evaluation: pipeline/runs/<model>/<behavior>/evaluations/reasoning_extended_evaluation.json

# Disable thinking mode (prefills <think></think>)
python -m pipeline.run_pipeline --model_path Qwen/Qwen3-8B --disable_thinking

## Visualization & Interactive Testing
# Visualize results
python visualize_results.py --run_dir pipeline/runs/Llama-2-7b-chat-hf/survival-instinct

# Completion Scripts
python -m run_completion --model_path Qwen/Qwen3-8B --enable_tools --max_tool_calls 1
python -m run_completion --model_path Qwen/Qwen3-8B --disable_thinking


# If running out of space, can run this for previous models
rm -rf /workspace/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf

# Clear model cache to free space
rm -rf /root/.cache/huggingface/hub/models--meta-llama--*
rm -rf /root/.cache/huggingface/hub/models--google--*