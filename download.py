# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: our gpt-j-6B model finetuned for title and teaser generation

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    print("downloading model checkpoint...")
    model = AutoModelForCausalLM.from_pretrained(
        "malteos/bloom-6b4-clp-german",
    )
    print("done")
    
    # load LoRA adapters finetuned for news snippet generation
    print("downloading adapter checkpoint")
    PeftModel.from_pretrained(model, "snipaid/snip-igel-500")
    print("done")

    # load tokenizer
    print("loading tokenizer...")
    AutoTokenizer.from_pretrained("philschmid/instruct-igel-001")
    print("done")

if __name__ == "__main__":
    download_model()
