unzip data_mmlu.zip
unzip data_mix.zip
python main.py --global_model 'llama-13b' --data_path  "./data_mmlu" --output_dir './nips-llama13b-full-mmlu-1-3-10/' --num_communication_rounds 1 --local_num_epochs 3 --full True
#python main.py --global_model 'llama-13b' --data_path  "./data_mix" --output_dir './nips-llama13b-full-mix-1-3-20/' --num_communication_rounds 1 --local_num_epochs 3 --full True --num_clients 20
