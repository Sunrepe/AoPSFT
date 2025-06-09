set -x

# please modify  --pretrain to your model path

read -r -d '' training_commands <<EOF
openrlhf.cli.train_mtsft \
   --max_len 8192 \
   --dataset data/sw/openrlhf_sftv4.json \
   --input_key conversation \
   --multiturn \
   --apply_chat_template \
   --train_batch_size 96 \
   --micro_train_batch_size 4 \
   --max_samples 5000 \
   --pretrain meta-llama/Llama-3.1-8B-Instruct \
   --save_path ./checkpoint/Llama-3.1-8B-Instruct/aopsftv0 \
   --save_steps -1 \
   --logging_steps 2 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 12 \
   --gradient_checkpointing \
   --bf16 \
   --flash_attn \
   --learning_rate 1.0e-5 \
   --adam_offload
EOF

if [[ ${1} != "slurm" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --module $training_commands
fi

# bash training/scripts/train_mtsft.sh