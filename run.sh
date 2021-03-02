export CUDA_VISIBLE_DEVICES=0

ambig_train=/home/sichenglei/reason_paths/graph_retriever/data/ambig_train.json
ambig_dev=/home/sichenglei/reason_paths/graph_retriever/data/ambig_dev.json
nq_train=/data3/private/clsi/AmbigQA/data/nqopen/train.json
nq_dev=/data3/private/clsi/AmbigQA/data/nqopen/dev.json
nq_test=/data3/private/clsi/AmbigQA/data/nqopen/test.json
dpr_data_dir=/data3/private/clsi/DPR


python3 run_graph_retriever.py \
--task ambigqa \
--bert_model bert-base-uncased --do_lower_case \
--train_file_path $ambig_train --dev_file_path $ambig_dev  \
--output_dir out/ \
--max_para_num 10 \
--neg_chunk 4 --train_batch_size 1 --gradient_accumulation_steps 1 \
--learning_rate 3e-5 --num_train_epochs 3 \
--max_select_num 11 #+1 for EOE
# --example_limit 5 \
