import json
import random
import string

n_var = 3 # number of variables
n_prompts = 600 # number of statements

def generate_random_data(n_var):
    variables = random.sample(string.ascii_lowercase, n_var)
    values = random.sample(range(100), n_var)
    q = random.randint(0, len(variables) - 1)
    q_value = values[q]
    q_var = variables[q]
    prompt = ''
    for i,(var,val) in enumerate(zip(variables,values)):
        if i > 0:
            prompt += f", {var}={val}"
        else:
            prompt += f"{var}={val}"
    prompt += f". {q_var}={q_value}."
    return "{"+prompt+"}\n"

def main(n_var,n_prompts):
    json_data = ''
    for _ in range(n_prompts):
        data = generate_random_data(n_var)
        json_data += data

    with open(f'{n_var}.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

if __name__ == "__main__":
    main(n_var,n_prompts)
