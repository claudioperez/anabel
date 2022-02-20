import yaml


def parseYamlFile(FileName, gc={}, lc={},**kwds):
    with open(FileName,'r') as f: file_str = f.read()

    compiled_str = compile('fr"""'+file_str+'"""', '<file_str>', 'eval')

    formated_str = eval(compiled_str, {**gc, **kwds}, lc)

    return yaml.load(formated_str, Loader=yaml.Loader)

if __name__=='__main__':

    print(parseYamlFile('test.yaml', gc={"L":20}, E=29e3))
