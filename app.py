from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

import torch

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    # load model
    print("loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "malteos/bloom-6b4-clp-german",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map='auto'
    )
    print("done")

    """# conditionally load model to GPU
    if device == "cuda:0":
        print("loading model to GPU...")
        model.cuda()
        print("done")"""
    
    # load LoRA adapters finetuned for news snippet generation
    print("downloading adapter checkpoint")
    PeftModel.from_pretrained(model, "snipaid/snip-igel-500")
    print("done")

    # load tokenizer
    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("philschmid/instruct-igel-001")
    print("done")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    document = model_inputs.pop("document", None)
    task_prefix = model_inputs.pop("task_prefix", None)
    prompt = model_inputs.pop('prompt', None)
    params = model_inputs.pop('params', None)
    
    # Handle missing arguments
    if document == None:
        return {'message': "No document provided"}

    if task_prefix == None:
        task_prefix = ""

    if prompt == None:
        return {'message': "No prompt provided"}

    if params == None:
        params = {}

    # Config
    generation_config = GenerationConfig(
        top_p=0.9,
        top_k=0,
        temperature=1,
        do_sample=True,
        early_stopping=True,
        length_penalty=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        **params
    )

    # Assemble prompt
    prompt = generate_prompt(prompt, document)

    # Embed prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    # Generate
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True
    )

    # Decode and return the result as a dictionary
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        output_text = output.split("### Antwort:")[1].strip()
        result = {"output": output_text}
        return result

def generate_prompt(instruction, input=None) -> str:
    if input:
        return f"""Nachfolgend finden Sie eine Anweisung, die eine Aufgabe beschreibt, und eine Eingabe, die weiteren Kontext liefert. Schreiben Sie eine Antwort, die die Aufgabe angemessen erfüllt.

### Anweisung:
{instruction}

### Eingabe:
{input}

### Antwort:"""
    else:
        return f"""Nachfolgend finden Sie eine Anweisung, die eine Aufgabe beschreibt. Schreiben Sie eine Antwort, die die Aufgabe angemessen erfüllt.

### Anweisung:
{instruction}

### Antwort:"""

