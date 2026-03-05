"""
DREAMER Struct Inspector
========================
Run this FIRST to see the exact structure of your DREAMER.mat file.
Place this in the same folder as DREAMER.mat and run:
    python inspect_dreamer.py
"""

import scipy.io as sio
import numpy as np

MAT_FILE = "C:/Users/Prajwal/Documents/Brain-Signaling/brain/data/emotion/DREAMER.mat"   # ← change if needed


def inspect_struct(obj, name="root", depth=0, max_depth=6):
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}[...] {name}  (max depth)")
        return

    if hasattr(obj, '_fieldnames'):
        print(f"{indent}[STRUCT] {name}")
        print(f"{indent}         fields = {obj._fieldnames}")
        for field in obj._fieldnames:
            child = getattr(obj, field, None)
            inspect_struct(child, name=f"{name}.{field}", depth=depth + 1, max_depth=max_depth)

    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            print(f"{indent}[CELL]   {name}  shape={obj.shape}")
            if obj.size > 0:
                first = obj.flat[0]
                inspect_struct(first, name=f"{name}[0]", depth=depth + 1, max_depth=max_depth)
        else:
            print(f"{indent}[ARRAY]  {name}  shape={obj.shape}  dtype={obj.dtype}")

    else:
        print(f"{indent}[VALUE]  {name}  type={type(obj).__name__}  = {repr(obj)[:100]}")


print("=" * 60)
print("  DREAMER.mat  Structure Inspector")
print("=" * 60)

mat     = sio.loadmat(MAT_FILE, struct_as_record=False, squeeze_me=True)
top_keys = [k for k in mat.keys() if not k.startswith('_')]
print(f"\nTop-level keys: {top_keys}\n")

dreamer = mat['DREAMER']
inspect_struct(dreamer, name="DREAMER", max_depth=7)

print("\n" + "=" * 60)
print("  Paste the output above when reporting issues.")
print("=" * 60)