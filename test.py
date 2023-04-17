# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = {
    'task_prefix': 'Your task is to generate a headline for a given text. \nText:',
    'document': 'Insert some news article fulltext here',
    'prompt': '\nHeadline: ',
    'params': {
        'min_new_tokens': 5,
        'max_new_tokens': 25
    }
}

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())