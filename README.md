# AL4RAG
# For model fine-tuning, run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes 1 --nproc_per_node 8 sft.py \
--model_name_or_path your_model_path \
--output_dir your_output_path \
--do_train \
--dataset detect_yesno \
--num_train_epochs 1 \
--learning_rate 1e-5 \
--drop_neg_ratio -1 \
--train_file your_training_set \
--eval_file your_eval_set \
--use_peft True \
--use_flashatt_2 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--model_max_length 4096 \
--report_to wandb \
--ddp_find_unused_parameters False \
--logging_steps 1 \
--run_name baseline \
--lr_scheduler_type 'cosine' \
--warmup_ratio 0.1 \
--save_steps 100 \
--save_total_limit 2 \
--overwrite_output_dir \
--eval_strategy steps \
--eval_steps 8 \
--fsdp "shard_grad_op auto_wrap" \
--fsdp_config fsdp.json
# For data selection, modify the file path and data proportion in AL4RAG.py, and run
python AL4RAG.py
# For DPO training, modify the eval set path in trainDPO.py, and run
python trainDPO.py \
    --dataset_name path_to_your_training_set \
    --model_name_or_path path_to_your_base_model \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --output_dir your_output_dir \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --eval_strategy steps \
    --eval_steps 50
# For answer generation, modify the model path and dataset path in generate_answers.py, and run
python generate_answers.py
# For evaluation process, please follow our paper.
