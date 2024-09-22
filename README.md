# FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations
Code of paper : [FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations](https://arxiv.org/pdf/2409.05976).

You can use this code to fine-tune LLMs with LoRA by WizardLLM dataset or other datasets.
The LoRA fine-tuning method includes FLoRA, FedIT, and Zero-Padding. You can also use heterogeneous LoRA rank settings in FLoRA and Zero-Padding.

## Requirments
Install all the packages from requirments.txt
* pip install -r requirements.txt
* git clone https://github.com/EleutherAI/lm-evaluation-harness
* cd lm-evaluation-harness
* pip install -e .

## Data
* The training dataset of WizardLLM has already been downloaded and split in ./data_wiz/ fold.
* If you want to use your dataset, use the same format as ./data_wiz/.

## Running the experiments
* To run the FLoRA algorithm (--stacking: True) and FedIT (--stacking False) in a homogeneous LoRA setting:
```
python main.py --global_model 'huggyllama/llama-7b' --data_path  "./data_wiz" --output_dir './FloRA-llama7b-wiz-homo/' --num_communication_rounds 3 --local_num_epochs 1 --stacking True
python main.py --global_model 'huggyllama/llama-7b' --data_path  "./data_wiz" --output_dir './FedIT-llama7b-wiz-homo/' --num_communication_rounds 3 --local_num_epochs 1 --stacking False
```
* To run the FLoRA algorithm (--stacking: True) and Zero-Padding (--stacking False --zero_padding True) in a heterogeneous LoRA setting:
```
python main.py --global_model 'huggyllama/llama-7b' --data_path  "./data_wiz" --output_dir './FloRA-llama7b-wiz-heter/' --num_communication_rounds 3 --local_num_epochs 1 --stacking True --heter True
python main.py --global_model 'huggyllama/llama-7b' --data_path  "./data_wiz" --output_dir './FedIT-llama7b-wiz-heter/' --num_communication_rounds 3 --local_num_epochs 1 --stacking False --heter True --zero_padding True
```

* To evaluate on LLM harness, try:
```
lm_eval --model_args pretrained=./FloRA-llama7b-wiz-homo/,parallelize=True,load_in_4bit=False, --tasks mmlu --num_fewshot 5 --batch_size 16 --output_path ../FloRA-llama7b-wiz-homo/
```
* To evaluate on MT-Bench, please follow the instructions on their websites: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
-----
