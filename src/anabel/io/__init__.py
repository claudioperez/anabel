import os

from anabel import COMPILER 

compiler = COMPILER


try:
    from ruamel.yaml import YAML
except ImportError:
    yaml = None
else: 
    yaml = YAML(typ='safe')


def _yaml_load(filepath):
    with open(filepath) as f: model_dict = yaml.load(f)
    return model_dict

def load(filepath):
    """Load from a serialized data file"""
    _, file_ext = os.path.splitext(filepath)
    
    if file_ext in ['.yml', '.yaml']: 
        return _yaml_load(filepath)

def dump(filepath): pass

def compile(func,inputs):
    if compiler is not None: 
        pass

