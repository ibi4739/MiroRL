# run on 8xH100
# make sure your current working directory is the root of the project
# Training with Musique + GAIA text-103 for train/test

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/mirorl/recipe/mcp/config"
TOOL_CONFIG_PATH="$PROJECT_DIR/mirorl/recipe/mcp/config/tool_config/mcp_tool_config.yaml"
MODEL_PATH=/your/model/path
DATA_HOME=/your/data/path
EXPERIMENT_NAME="mirorl-14b_genqa_64k"


python -m mirorl.recipe.mcp.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='mcp_trainer' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=64 \
    data.max_prompt_length=3072 \
    data.max_response_length=62464 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.enable_tokenization_sanity_check=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.3 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG_PATH \
    actor_rollout_ref.rollout.multi_turn.max_turns=40 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='odr-mcp' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=$PROJECT_DIR/train_rollout_data/$EXPERIMENT_NAME \
    trainer.validation_data_dir=$PROJECT_DIR/val_rollout_data/$EXPERIMENT_NAME \
    data.train_files=$DATA_HOME/genqa/train.parquet \
    data.val_files=$DATA_HOME/genqa/val.parquet \
    trainer.total_epochs=15 $@
