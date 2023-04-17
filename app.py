from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    # load model
    print("loading base model to CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        "malteos/bloom-6b4-clp-german",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map='auto'
    )
    print("done")

    # conditionally load model to GPU
    if device == "cuda:0":
        print("loading model to GPU...")
        model.cuda()
        print("done")
    
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

    # Initialize pipeline
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, **params)

    # Run generation pipline
    output = gen_pipe(f"{task_prefix} {document} {prompt}")

    # Get output text
    output_text = output[0]['generated_text'].split(prompt)[1].split("</s>")[0]

    # Return the results as a dictionary
    result = {"output": output_text}
    return result
