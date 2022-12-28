import os
from pathlib import Path
import subprocess

cwd = os.getcwd()
cwd_p = Path(cwd)

inputs_p = cwd_p / 'inputs'
outputs_p = cwd_p / 'outputs'


exts = [".blend", ".BLEND"]
input_files = [f for f in inputs_p.glob('*') if f.suffix in exts]

for blend_file_p in input_files[::2]:
    print(blend_file_p)
    output_blend = str(outputs_p) + '\\'+ blend_file_p.name 
    try:
        subprocess.run(["blender", str(blend_file_p), "--python",  cwd + "\\bake.py",  "--", "-s",str(output_blend), '-p', str(cwd) ], check=True)
    
    except Exception as e:
        print("file ", blend_file_p.name, " failed due to error: ", e)
    
