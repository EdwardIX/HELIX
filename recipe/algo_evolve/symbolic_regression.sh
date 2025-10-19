set -x

# Check and set environment variables with defaults
if [ -z "$TRAINER_NNODES" ]; then
  TRAINER_NNODES=1
fi
if [ -z "$TRAINER_NGPUS_PER_NODE" ]; then
  TRAINER_NGPUS_PER_NODE=8
fi
echo "Using nnodes=$TRAINER_NNODES, gpus_per_node=$TRAINER_NGPUS_PER_NODE"

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

######### Input Experiment Name #########
if [ -n "$1" ]; then
  experiment_name=$1
else
  read -r -p "Name of this experiment (default: None):" experiment_name
fi
######### Training Configuration #########
total_epochs=150 # Total training epochs
# total_epochs=50 # Temporarily set to 50 for param scaning
# save_frequency=50 # Save model
save_frequency=0 # Temporarily set to 0

# Lora Configs
lora_rank=0 # 0 to disable lora
lora_alpha=$((lora_rank * 2))
lora_target_modules=all-linear

# Learning Rate
if [ "$lora_rank" -gt 0 ] ; then
  learning_rate=1e-5
  rollout_load_format=safetensors
else
  learning_rate=1e-6
  rollout_load_format=dummy_dtensor
fi
# Loss Config
kl_loss_coef=0.001
######### Batchsize and sample size #########
dataloader_batch_size=16 # Global batch size for all gpus. Indicates number of **Experiences** and **will be multiplied by grpo_sample_size** to be the real GLOBAL training batchsize. 
ppo_mini_batch_size=16 # Global batch size for all gpus. In one step, there will be **Consecutive** (dataloader_batch_size / ppo_mini_batch_size) param updates.
use_dynamic_bsz=True
micro_batch_size_per_gpu=1 # If don't use dynamic batch size, this is the micro batch size per gpu.
actor_max_token_len_per_gpu=16384 # If using dynamic batch size, the following 3 configs is the max token length per gpu for actor rollout.
rollout_max_token_len_per_gpu=$actor_max_token_len_per_gpu
ref_max_token_len_per_gpu=65536
grpo_sample_size=16
######### Prompt and Response Length #########
max_prompt_length=10240 # Maximum prompt length
max_response_length=6144 # Maximum response length
rollout_max_num_batched_tokens=$((max_prompt_length + max_response_length)) 
######### Sampling parameters ##########
top_p=0.95
temperature=1.0
######### Validation Configuration #########
# test_frequency=20 # 0 for no validation
test_frequency=0
# val_before_train=True
val_before_train=False
do_sample_in_val=True
if [ "$do_sample_in_val" = "True" ] ; then
  validation_rollouts=64 # Number of rollouts to use for validation
  val_top_p=$top_p
  val_temperature=$temperature
else
  validation_rollouts=1
  val_top_p=1.0
  val_temperature=0.0
fi
######### Names and Paths ##########
project_name='math_symbolic_regression_evolve'
model_name_or_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
n_gpus_per_node=4
# model_name_or_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
# model_name_or_path='Qwen/Qwen3-14B'
if [ -z "${experiment_name}" ]; then
  experiment_name="$(date '+%Y-%m-%d_%H:%M:%S')"
else
  experiment_name="$(date '+%Y-%m-%d_%H:%M:%S')_${experiment_name}"
fi
project_base_path=$(realpath ../../data)/models/${project_name}/${experiment_name}
train_datapath=$(realpath ../../data)/datasets/verl/math_program_symbolic_regression/train.parquet
test_datapath=$(realpath ../../data)/datasets/verl/math_program_symbolic_regression/test.parquet
train_files="['$train_datapath']"
test_files="['$test_datapath']"
######### Evolve Algorithm Configuration #########
dynamic_prompt_sampler_max_history=10 # Number of previous trials
dynamic_sample_strategy=NSGA

mkdir -p $project_base_path

# Set PYTHONPATH with fallback
if [ -z "$PYTHONPATH" ]; then
  python_path="../../"
else
  python_path="$PYTHONPATH"
fi

PYTHONPATH=$python_path python3 -m recipe.algo_evolve.main_evolve \
    algorithm.adv_estimator=grpo \
    data.custom_cls.path=./dataset/grouped_dynamic_rl_dataset.py \
    data.custom_cls.name=GroupedDynamicRLHFDataset \
    +data.custom_cls.init_kwargs.log_path=$project_base_path/evolve_dataset_logs \
    +data.custom_cls.init_kwargs.language=python \
    +data.custom_cls.init_kwargs.max_trial_per_ds=1000 \
    +data.custom_cls.init_kwargs.max_bsz_per_ds=8 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$dataloader_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.dynamic_prompt_sampler.max_history_length=$dynamic_prompt_sampler_max_history \
    +data.dynamic_sample_strategy.type=$dynamic_sample_strategy \
    actor_rollout_ref.model.path=$model_name_or_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=$lora_rank \
    actor_rollout_ref.model.lora_alpha=$lora_alpha \
    actor_rollout_ref.model.target_modules=$lora_target_modules \
    actor_rollout_ref.actor.optim.lr=$learning_rate \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$rollout_max_token_len_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$grpo_sample_size \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_max_num_batched_tokens \
    actor_rollout_ref.rollout.load_format=$rollout_load_format \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$do_sample_in_val \
    actor_rollout_ref.rollout.val_kwargs.n=$validation_rollouts \
    actor_rollout_ref.rollout.val_kwargs.top_p=$val_top_p \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$ref_max_token_len_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$TRAINER_NGPUS_PER_NODE \
    trainer.nnodes=$TRAINER_NNODES \
    trainer.save_freq=$save_frequency \
    trainer.test_freq=$test_frequency \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=$val_before_train \
    trainer.default_local_dir=$project_base_path \
    trainer.rollout_data_dir="${project_base_path}/train_rollouts" \
    trainer.validation_data_dir="${project_base_path}/validation_rollouts" \
    reward_model.reward_manager=custom_batch \
    custom_reward_function.path=../../src/rewards/reward.py \
    custom_reward_function.name=custom_reward_fn 2>&1 | tee $project_base_path/output.log