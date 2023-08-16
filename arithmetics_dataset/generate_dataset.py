import json
import random
import string

n_var = 3 # number of variables
n_prompts = 600 # number of statements

def generate_random_data(n_var,existing_data):
    while True:
        variables = random.sample(string.ascii_lowercase, n_var)
        values = random.sample(range(100), n_var)
        q = random.randint(0, len(variables) - 1)
        q_value = values[q]
        q_var = variables[q]
        data = {f"var_{i}": var for i, var in enumerate(variables)}
        data.update({f"val_{i}": val for i, val in enumerate(values)})
        frozen_data = frozenset(data.items())  # Convert the dictionary to a frozenset
        if not (frozen_data in existing_data):  # Add the new data to the set
            break

    existing_data.add(frozen_data)    
    prompt = ''
    for i,(var,val) in enumerate(zip(variables,values)):
        if i > 0:
            prompt += f", {var}={val}"
        else:
            prompt += f"{var}={val}"
    prompt += f". {q_var}={q_value}."
    return prompt

def main(n_var,n_prompts):
    json_data = []
    existing_data = set()
    for _ in range(n_prompts):
        prompt = generate_random_data(n_var,existing_data)
        json_data.append({"sentence": prompt})

    with open(f'{n_var}.json', 'w') as json_file:
        for data in json_data:
            json.dump(data, json_file)
            json_file.write('\n')
if __name__ == "__main__":
    main(n_var,n_prompts)
