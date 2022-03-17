import glob
import os
import pathlib

from nbconvert import NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import pytest
from traitlets.config import Config


def run_notebook(path, timeout=120):
    """
    Execute a Jupyter Notebook and return any errors that it produces.

    :param path: path to the .ipynb file to execute
    :param int timeout: number of seconds to let the notebook run before we throw an exception by default.
    :return: a tuple (nb, errors) containing the parsed Notebook object and a list of errors, respectively
    """
    # We'll use a NotebookExporter from the nbconvert package to load the notebook and re-export it to a temporary file.
    # First we want to configure it to execute the notebook before writing it:
    c = Config()
    c.NotebookExporter.preprocessors = ['nbconvert.preprocessors.ExecutePreprocessor']
    c.ExecutePreprocessor.timeout = timeout
    c.ExecutePreprocessor.kernel_name = 'LEAP_venv'  # We assume a kernel named "LEAP_venv" that lives in our venv
    exp = NotebookExporter(config=c)

    # Load the notebook
    with open(path, 'r') as nb_file:
        body, resources = exp.from_file(nb_file)

    # Parse the notebook string into a notebook object
    nb = nbformat.reads(body, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]

    return nb, errors


def is_hidden_path(path):
    """Return true if the file or any of its ancesor directories begins with '.'."""
    head, tail = os.path.split(path)
    if tail.startswith('.'):
        return True
    elif head == '' or tail == '':
        return False
    else:
        return is_hidden_path(head)


notebooks = pathlib.Path(__file__, '..', '../..', 'examples').resolve().rglob('*.ipynb')
notebooks = [ p for p in notebooks if not is_hidden_path(p) ]


# We give Jupyter tests a separate marker, because they can only run if the 'LEAP_venv' kernel is configured propertly by the user
@pytest.mark.jupyter
@pytest.mark.parametrize('path', notebooks)
def test_notebook(path):
    """Ensure that all of the notebooks in the examples directory run without errors."""
    nb, errors = run_notebook(path)

    # No errors is success
    assert errors == []
