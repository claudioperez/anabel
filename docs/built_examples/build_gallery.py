"""Convert jupyter notebook to sphinx gallery notebook styled examples.
Usage: python ipynb_to_gallery.py <notebook.ipynb>
Dependencies:
pypandoc: install using `pip install pypandoc`
"""

import pypandoc as pdoc
import json

# "C:\Users\claud\OneDrive\400_box\Python\myPackages\ema\docs>"
# "python ./built_examples/build_gallery.py"
def convert_ipynb_to_gallery(file_name):
    python_file = ""

    nb_dict = json.load(open(file_name))
    cells = nb_dict['cells']
    

    for i, cell in enumerate(cells):
        if i == 0:  
            try:
                assert cell['cell_type'] == 'markdown', \
                    'First cell has to be markdown'
            except:
                return

            md_source = ''.join(cell['source'])
            rst_source = pdoc.convert_text(md_source, 'rst', 'md')
            python_file = '"""\n' + rst_source + '\n"""'
        else:
            if cell['cell_type'] == 'markdown':
                md_source = ''.join(cell['source'])
                rst_source = pdoc.convert_text(md_source, 'rst', 'md')
                commented_source = '\n'.join(['# ' + x for x in
                                              rst_source.split('\n')])
                python_file = python_file + '\n\n\n' + '#' * 70 + '\n' + \
                    commented_source
            elif cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                python_file = python_file + '\n' * 2 + source

    python_file = python_file.replace("\n%", "\n# %")

    open(file_name.replace('.ipynb', '.py'), 'w').write(python_file)

if __name__ == '__main__':
    import os
    galleries = ['structuralAnalysis', 'plasticanalysis', 'dynamics']
    
    for gallery in galleries:    
        ex_directory = f'../examples/'+ gallery
        blt_directory = '.'

        for filename in os.listdir(ex_directory):
            if filename.endswith(".ipynb"): 
                ex_file = os.path.join(ex_directory, filename)
                print(ex_file)
                convert_ipynb_to_gallery(ex_file)