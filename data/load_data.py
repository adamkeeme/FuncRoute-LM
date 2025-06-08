import json
import random

import json

SYSTEM_PROMPT = '''Given a function catalog and NATURAL language user request. Your goal is to call the TARGET function.
Then output ONLY a JSON object with keys:
  - "user_request" : the request string
  - "call": { "name": <function name>, "arguments": <JSON arguments> }
No extra text. Make sure arguments match the parameter types.
'''

def load_random_functions(filepath: str):

    with open(filepath, "r", encoding="utf-8") as f:
        functions = json.load(f)

    for func in functions:
        func.pop("code", None)  # 安全删除，若无 'code' 键不会报错
    n = random.randint(3, 8)
    return random.sample(functions, n)

def find_function_by_name(file_path: str, target_name: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        functions = json.load(f)

    for func in functions:
        if func.get("name") == target_name:
            return func
    return None


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]



def load_train_data_in_query_anwer_pair(how_many, train_dataset_file, functions_with_params_file): # return List[tuple(query, answer)]
    train_data_with_gt_function_call_file_path = train_dataset_file
    train_data_with_gt_function_call = load_jsonl(train_data_with_gt_function_call_file_path)

    # print(data[0])
    # print("\n\nuser query\n", data[0]["user_request"])
    # print("\n\ncalled function\n", data[0]["call"]["name"])
    # print("\n\nground truth answer\n", data[0]["call"])
    train_set = []

    for i, d in enumerate(train_data_with_gt_function_call[:how_many]):
        print("\n\n\ndata counter: ", i)
        selected_funcs = load_random_functions(functions_with_params_file)
        # print(selected_funcs)

        current_called_function = find_function_by_name(functions_with_params_file, d["call"]["name"])
        current_called_function.pop("code", None)
        # print(current_called_function)

        selected_funcs.append(current_called_function)
        # print("\n\n")
        # print(selected_funcs)
        random.shuffle(selected_funcs)
        # print(selected_funcs)

        input_prompt = SYSTEM_PROMPT + "\nFunction catalog:\n"

        for fc in selected_funcs:
            input_prompt += json.dumps(fc, ensure_ascii=False) + "\n"

        input_prompt += "\nQuery:\n" +  d["user_request"]
        input_prompt += "\n\nAnswer:"

        print(input_prompt)

        gt_answer = d["call"]

        print(gt_answer)
        train_set.append((input_prompt, gt_answer))
    
    return train_set



dataset = load_train_data_in_query_anwer_pair(1, "D:/shawn_workspace/MSAI337_NLP/train_dataset.jsonl", "D:/shawn_workspace/MSAI337_NLP/functions_with_params.json")
print(dataset)



'''Example:

Given a function catalog and NATURAL language user request. Your goal is to call the TARGET function.
Then output ONLY a JSON object with keys:
  - "user_request" : the request string
  - "call": { "name": <function name>, "arguments": <JSON arguments> }
No extra text. Make sure arguments match the parameter types.

Function catalog:
{"name": "solution", "description": "Considering any range can be provided,\nbecause as per the problem, the digit d < 1000\n>>> solution(1, 10)\n7\n>>> solution(10, 100)\n97\n>>> solution(10, 1000)\n983", "parameters": {"numerator": "int – 1", "digit": "int – 1000"}}
{"name": "make_lowpass", "description": "Creates a low-pass filter\n\n>>> filter = make_lowpass(1000, 48000)\n>>> filter.a_coeffs + filter.b_coeffs  # doctest: +NORMALIZE_WHITESPACE\n[1.0922959556412573, -1.9828897227476208, 0.9077040443587427, 0.004277569313094809,\n 0.008555138626189618, 0.004277569313094809]", "parameters": {"frequency": "int", "samplerate": "int", "q_factor": "float – 1 / sqrt(2)"}}
{"name": "solution", "description": "Returns the largest palindrome made from the product of two 3-digit\nnumbers which is less than n.\n\n>>> solution(20000)\n19591\n>>> solution(30000)\n29992\n>>> solution(40000)\n39893\n>>> solution(10000)\nTraceback (most recent call last):\n    ...\nValueError: That number is larger than our acceptable range.", "parameters": {"n": "int – 998001"}}
{"name": "encode", "description": ">>> encode(\"myname\")\n[13, 25, 14, 1, 13, 5]", "parameters": {"plain": "str"}}

Query:
I need to create a lowpass filter with a cutoff frequency of 5000 Hz and a sampling rate of 44100 Hz. Can you help generate that?

Answer:
{'name': 'make_lowpass', 'arguments': {'frequency': 5000, 'samplerate': 44100, 'q_factor': 0.7071067811865476}}


'''
