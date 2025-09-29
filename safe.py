from safetensors import safe_open
from glob import glob

import natsort

tensors = {}

def process_tensor(name, tensor):
    tensors[name] = tensor
    print(f"{name}: ({tensor.shape} at {tensor.dtype})")

filenames = "/tmp/Qwen3-32B/model*.safetensors"
filenames = sorted(glob(filenames), key=natsort.natsort_key)
for fname in filenames:
    with safe_open(fname, framework="pt") as fn:
        sorted_keys = fn.keys()
        assert all(isinstance(key, str) for key in sorted_keys), "All keys must be strings"
        for key in sorted_keys:
            process_tensor(key, fn.get_tensor(key))

print(f"Read {len(tensors)} tensors.")
