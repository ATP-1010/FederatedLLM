python main.py --global_model 'google/gemma-7b' --data_path  "./data_wiz" --output_dir './nips-gemma7b-full-wiz-1-3-10/' --num_communication_rounds 1 --local_num_epochs 3 --full True
python main.py --global_model 'google/gemma-7b' --data_path  "./data_mmlu" --output_dir './nips-gemma7b-full-mmlu-1-3-10/' --num_communication_rounds 1 --local_num_epochs 3 --full True
python main.py --global_model 'google/gemma-7b' --data_path  "./data_mix" --output_dir './nips-gemma7b-full-mix-1-3-10/' --num_communication_rounds 1 --local_num_epochs 3 --full True --num_clients 20

python main.py --global_model 'google/gemma-7b' --data_path  "./data_wiz" --output_dir './nips-gemma7b-full-wiz-3-1-10/' --num_communication_rounds 3 --local_num_epochs 1 --full True
python main.py --global_model 'google/gemma-7b' --data_path  "./data_mmlu" --output_dir './nips-gemma7b-full-mmlu-3-1-10/' --num_communication_rounds 3 --local_num_epochs 1 --full True
python main.py --global_model 'google/gemma-7b' --data_path  "./data_mix" --output_dir './nips-gemma7b-full-mix-3-1-20/' --num_communication_rounds 3 --local_num_epochs 1 --full True --num_clients 20
