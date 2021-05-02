import subprocess
import tempfile

import pytest

def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000",
                "--output", fout.name, path]
        subprocess.check_call(args)

@pytest.mark.parametrize("nb",["elle-0020"])
def test(nb):
    _exec_notebook(f"tests/{nb}.ipynb")

