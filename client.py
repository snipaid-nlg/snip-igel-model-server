import banana_dev as banana
import os

# This is some examplary code on how to call the model server, when it is hosted at banana.dev

api_key = os.environ.get("API_KEY")
model_key = os.environ.get("MODEL_KEY")

model_inputs = {
    'task_prefix': 'Your task is to generate a headline for a given text. \nText:',
    'document': 'Insert some news article fulltext here',
    'prompt': '\nHeadline: ',
    'params': {
        'min_new_tokens': 5,
        'max_new_tokens': 25
    }
}

model_outputs = banana.run(api_key, model_key, model_inputs)

print(model_outputs)