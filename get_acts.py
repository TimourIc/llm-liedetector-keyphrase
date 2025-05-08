import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM)
import yaml
import numpy as np
import os
import pickle
from argparse import ArgumentParser
import pandas as pd
from transformers.utils.logging import disable_progress_bar
import gc
disable_progress_bar()

MODELS={"Llama3-8b-instruct":"meta-llama/Meta-Llama-3-8B-Instruct","Mistral-8b-instruct":"mistralai/Ministral-8B-Instruct-2410"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pickles_proc(directory_path="data/proc"):

    """Reads in all pickle names and files in the directory path"""


    pickle_files = []
    pickle_names=[]

    for file in os.listdir(directory_path):
        if file.endswith(".pkl"):
            file_name = os.path.splitext(file)[0]
            pickle_names.append(file_name)
            file_path = os.path.join(directory_path, file)
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
                pickle_files.append((file_name, obj))

    return pickle_names, pickle_files

class Hook:
    
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        
        self.out= module_outputs[0]

def get_acts(
        dataset,
        tokenizer,
        model,
        layers=[12]):
    """
    Collects the activations in the last hidden state for the layers provided in the list
    """

    acts = {l: [] for l in layers}

    def _make_hook(layer_id):
        def _hook(_module, _inp, out):
            capture = out[0][0, -1, :].detach().cpu()
            acts[layer_id].append(capture)


        return _hook

    handles = [
        model.model.layers[l].register_forward_hook(_make_hook(l))
        for l in layers
    ]

    labels = []
    with torch.no_grad():
        for _, row in dataset.iterrows():
            input_ids = tokenizer.encode(row.prompt, return_tensors="pt").to(device)
            _ = model(input_ids, use_cache=False)               
            labels.append(row.label)

            del input_ids, _
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    acts   = {k: torch.stack(v) for k, v in acts.items()}
    labels = torch.tensor(np.array(labels))

    return acts, labels

if __name__=="__main__":
    

    pickle_names, pickle_files= get_pickles_proc()

    parser=ArgumentParser()
    parser.add_argument("--HF_TOKEN", type=str, default="None",  help=f"HuggingFace token with persmissions to download models in MODELS")
    parser.add_argument("--LAYERS", nargs='+', type=int, default=[12,14,16,18,20],  help=f"Layer to get the activations from")
    parser.add_argument("--FORMATS", nargs='+', type=str, default=pickle_names, help="List of formats to get acts for (sometimes you only want one)")
    parser.add_argument("--MODELS", nargs='+', type=str, default=MODELS.keys)
   
    args = parser.parse_args()

    HF_token=args.HF_TOKEN
    if args.HF_TOKEN=="None":
        with open("config/keys.yaml") as f:
            config_dict=yaml.safe_load(f )
        HF_token=config_dict["HF_token"]


    for model_name_short in args.MODELS:

        model_name=MODELS[model_name_short]

        print(f"Loading {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_token)
        model.to(device)
        model.eval() 

        for pickle_name in args.FORMATS:

            print(f"Getting {model_name} activations for {pickle_name} class")


            with open(f"data/proc/{pickle_name}.pkl","rb") as f:
                datasets=pickle.load(f)
                
            for ds_name,dataset in datasets.items():

                acts,labels=get_acts(dataset,tokenizer,model, args.LAYERS)
                for layer in acts.keys():

                    save_dict={"acts":acts[layer],"labels":labels}
                    directory=f"data/acts/{model_name_short}/layer_{layer}/{pickle_name}"
                    os.makedirs(directory, exist_ok=True)
                    torch.save(save_dict, f"{directory}/{ds_name}.pt")

        del model, tokenizer
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
  