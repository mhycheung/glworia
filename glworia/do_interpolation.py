from jax.config import config
config.update("jax_enable_x64", True)

from glworia.interpolate import interpolate, interpolate_im
import argparse
import json

print('''

  ____ _                    _       
 / ___| |_      _____  _ __(_) __ _ 
| |  _| \ \ /\ / / _ \| '__| |/ _` |
| |_| | |\ V  V / (_) | |  | | (_| |
 \____|_| \_/\_/ \___/|_|  |_|\__,_|

''')

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--input-file', dest = 'input_file', type=str, required=True)
parser.add_argument('-s', '--save-dir', dest = 'save_dir', type=str, required=True)
parser.add_argument('-im', '--image', dest = 'image', action='store_true')
args = parser.parse_args()

input_file = args.input_file
save_dir = args.save_dir
image = args.image

with open(input_file, 'r') as f:
    input_dict = json.load(f)

if image:
    interpolate_im(input_dict, save_dir)
interpolate(input_dict, save_dir)


