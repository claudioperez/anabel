import os

from anabel import COMPILER 
import anabel.io.parse

compiler = COMPILER


# try:
#     from ruamel.yaml import YAML
# except ImportError:
#     yaml = None
# else: 
#     yaml = YAML()
#     yaml.indent(mapping=2, sequence=4, offset=2)
import yaml

def _yaml_load(filepath):
    #with open(filepath) as f: model_dict = yaml.load(f) # ruamel.yaml
    with open(filepath) as f: model_dict = yaml.load(f,Loader=yaml.FullLoader) # PyYaml
    return model_dict

def load(filepath):
    """Load from a serialized data file"""
    _, file_ext = os.path.splitext(filepath)
    
    if file_ext in ['.yml', '.yaml']:
        return _yaml_load(filepath)

def dump_yaml(doc, fstream):
    yaml.dump(doc, fstream)

def compile(func,inputs):
    if compiler is not None: 
        pass

