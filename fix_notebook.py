import json

notebook_path = r"u:\assesment\notebooks\DeepGlobe_PartialCE.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "image_dir = os.path.join(DATA_ROOT, \"images\")" in line:
                new_source.append("if USE_SYNTHETIC_DATA:\n")
                new_source.append("    image_dir = os.path.join(DATA_ROOT, \"images\")\n")
                new_source.append("    mask_dir = os.path.join(DATA_ROOT, \"masks\")\n")
                new_source.append("else:\n")
                new_source.append("    image_dir = os.path.join(DATA_ROOT, \"train\")\n")
                new_source.append("    mask_dir = os.path.join(DATA_ROOT, \"train\")\n")
            elif "mask_dir = os.path.join(DATA_ROOT, \"masks\")" in line:
                pass # skip
            elif "image_files = sorted(os.listdir(image_dir))" in line:
                new_source.append("image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(\"_sat.jpg\")])\n")
            elif "mask_name = [f for f in os.listdir(mask_dir) if f.startswith(stem)][0]" in line:
                new_source.append("    mask_name = [f for f in os.listdir(mask_dir) if f.startswith(stem) and f.endswith(\"_mask.png\")][0]\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
