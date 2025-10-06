#!/usr/bin/python
"""analyse a given safetensors object to give relevant information"""

# imports
import argparse
import re
from pathlib import Path
from datetime import datetime
from safetensors import safe_open
import json

# local imports
from print_utils import print_error
from utils import decimal_to_base


def analyse_safetensors_file(filepath):
    """analyse a safetensors file"""

    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    stat_data = file_path.stat()

    file_size_mb = stat_data.st_size / (1024 * 1024)
    last_accessed_date = datetime.fromtimestamp(stat_data.st_atime)
    last_modified_date = datetime.fromtimestamp(stat_data.st_mtime)

    # this is stored in decimal, but is far more readable in octal
    # the data format is : {file_type}{permissions} where each is a 3 bit string
    file_mode = stat_data.st_mode
    permissions_code = decimal_to_base(file_mode % (8 ** 3), 8)
    file_type_code = decimal_to_base((file_mode - (file_mode % (8 ** 3))) // (8 ** 3), 8)

    print(f"File size: {file_size_mb:.2f}Mb")
    print(f"Last accessed: {last_accessed_date.strftime("%Y-%m-%d %H:%M:%S")}")
    print(f"Last modified: {last_modified_date.strftime("%Y-%m-%d %H:%M:%S")}")
    print("=============")
    print(f"File type code: {file_type_code}")
    print(f"File permissions code: {permissions_code}")
    print()


def scan_metadata(filepath):
    """analyse safetensors metadata"""
    
    with safe_open(filepath, framework="pt", device="cpu") as f:
        metadata = f.metadata()

        # parsing the main data strings
        parsed_data = {}
        
        for key, value in metadata.items():
            # print(f"{key} -> {value}")
            if value and (value.startswith('{') or value.startswith('}')):
                parsed_data[key] = json.loads(value)
            elif value and value.replace('.', '').replace('-', '').replace(',', '').isdigit():
                try:
                    parsed_data[key] = float(value) if "." in value else int(value)
                except Exception:
                    parsed_data[key] = str(value)
            elif value.lower() in ['true', 'false']:
                parsed_data[key] = value.lower() == 'true'
            elif value.lower() == 'none':
                parsed_data[key] = None
            else:
                parsed_data[key] = value

    print("Model Name:", parsed_data.get('ss_output_name'))
    print("Training Images:", parsed_data.get('ss_num_train_images'))
    print("Epochs:", parsed_data.get('ss_epoch'))
    print("Learning Rate:", parsed_data.get('ss_learning_rate'))

    # Access the tag frequencies
    if 'ss_tag_frequency' in parsed_data:
        tags = parsed_data['ss_tag_frequency']
        print("")
        print("Most common tags:")
        for category, tag_dict in tags.items():
            sorted_tags = sorted(tag_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"{category}: {sorted_tags}")

    # Access dataset info
    if 'ss_datasets' in parsed_data:
        datasets = parsed_data['ss_datasets']
        print("")
        print(f"Dataset info: {datasets}")



def parse_arguments():
    parser = argparse.ArgumentParser(description="analyse a safetensors object")
    parser.add_argument("file", help="path to safetensors file", type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()
    file = args.file

    if re.search(r".*(\.safetensors)", file).group(1).lower() != '.safetensors':
        print_error("Error: File input does not have the '.safetensors' file extension")

    analyse_safetensors_file(file)
    scan_metadata(file)


if __name__ == '__main__':
    main()
