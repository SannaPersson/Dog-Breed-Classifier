import torch
import json
import os
import sys

def save_checkpoint(state, filename="my_checkpoint.pt"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def trace_model(model, filename="traced_model.pt"):
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(filename)

def make_indices(root):
    # Loop
    index_to_names = {}
    idx = 0
    for (dir, subdirs, files) in os.walk(root):
        try:
            index_to_names[str(idx)] = " ".join(dir.split("-")[1].split("_"))
            idx+=1
        except IndexError:
            continue
    with open("index_to_names.json", "w") as f:
        json.dump(index_to_names, f)

