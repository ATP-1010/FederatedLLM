import os
from typing import List
from tqdm import tqdm
import fire
import torch
import datasets
from transformers import GenerationConfig
import json
import csv
from peft import set_peft_model_state_dict
import numpy as np
import random

model_type = 'llama'
datasets.utils.logging.set_verbosity_error()
device_map = "auto"
max_new_token: int = 32
verbose: bool = False

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1)

def global_evaluation(model, tokenizer, prompter, dev_data_path):
    data_class =  ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    right_count_dict = dict.fromkeys(data_class, 0)
    total_count_dict = dict.fromkeys(data_class, 0)
    acc_count_dict = dict.fromkeys(data_class, 0)
    with open(dev_data_path, 'r') as f:
        test_set = json.load(f)
    count=0

    if model_type == 'llama':
        sampling = GenerationConfig(
            do_sample=True,
            temperature=0.2,
            top_p=0.6,
            top_k=30,
            num_beams=1,
            max_new_tokens=max_new_token,
            early_stopping=True,
        )

    if model_type == 'gpt2':
        sampling = GenerationConfig(
            bos_token_id = 50256,
            eos_token_id = 50256,
            _from_model_config = True,
        )

    for data_point in tqdm(test_set):
        count +=1
        target = data_point["output"]
        class_test_set = data_point["class"]
        
        tgt_ans_idx = target.replace('The answer is: ','').split('. ')[0]
        tgt_ans = target.replace('The answer is: ','').split('. ')[1]

        test_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            'The answer is: ',
        )

        with torch.autocast("cuda"):
            inputs = tokenizer(test_prompt, return_tensors="pt")
            input =inputs["input_ids"].to('cuda')
            with torch.no_grad():
                #print(tokenizer.eos_token_id, tokenizer.pad_token_id)
                generation_output = model.generate(
                    input_ids=input,
                    generation_config=sampling,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_token,
                    pad_token_id=tokenizer.eos_token_id
                )
            generation_output_decoded = tokenizer.decode(generation_output.sequences[0])
            # print(generation_output_decoded)
            split = prompter.template["response_split"]
            ans = generation_output_decoded.split(split)[-1].strip()
            if verbose:
                print('-------------------')
                print(test_prompt)
                print(tgt_ans)
                print(tgt_ans_idx)
                print(ans)
            if tgt_ans_idx+'.' in ans or tgt_ans in ans:
            # if tgt_ans_idx in ans or tgt_ans in ans:
                right_count_dict[class_test_set] += 1
            total_count_dict[class_test_set] += 1

    mean_acc = 0.

    for key in acc_count_dict.keys():
        tmp = right_count_dict[key]/total_count_dict[key]
        mean_acc += tmp
        acc_count_dict[key] = tmp
    mean_acc /= len(acc_count_dict.keys())
    csv_data = [right_count_dict, total_count_dict, acc_count_dict]

    '''with open(os.path.join('/ai4bio-store/junbo.li/data_selection/alpaca-lora/raw_dict_mmlu',data_path.split('/')[-1].replace('.json','') + '.csv'), 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=right_count_dict.keys())
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)'''
    if verbose:
       print(right_count_dict)
    #print(total_count_dict)
    print('Acc: ', acc_count_dict)
    print()
    #score = eval_usmle(model, dev_data_path, tokenizer, verbose=False)
    print('========== Accuracy ==========')
    print(mean_acc)
    
    return mean_acc

#model = LlamaForCausalLM.from_pretrained(
#model = AutoModelForCausalLM.from_pretrained(
#tokenizer = LlamaTokenizer.from_pretrained('linhvu/decapoda-research-llama-7b-hf')
#tokenizer = AutoTokenizer.from_pretrained('gpt2')
#tokenizer.pad_token_id = tokenizer.eos_token_id
#print(tokenizer.pad_token_id, tokenizer.eos_token_id)
'''tokenizer.pad_token_id = (
    0
)
tokenizer.padding_side = "left"'''
# = Prompter("alpaca")

'''for id in range(1, 10):
    single_weights = torch.load('./lora-shepherd-7b-autolora-1-4/10/0/local_output_{}/'.format(id))
    set_peft_model_state_dict(model_c, single_weights, "default")
    for param_tensor in model_c.state_dict():
        model.state_dict[param_tensor] += model_c.state_dict[param_tensor]

for param_tensor in model.state_dict():
    model.state_dict[param_tensor] = model.state_dict[param_tensor]/10.0'''

'''with open(count_fine_path, "a") as file:
    file.write(str({"dataset_name": data_path.split('/')[-1], "accuracy": score})+'\n')'''