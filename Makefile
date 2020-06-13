help:
	@echo \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
	@echo \# LEAP Makefile
	@echo \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
	@echo \#
	@echo "#	make venv			Create a virtual environment"
	@echo "#	source venv/bin/activate	Activate it"
	@echo \#
	@echo "#	make setup			Install LEAP & dependencies"
	@echo "#	make doc			Build docs (in docs/build/html/)"
	@echo "#	make dist			Create package (i.e. for PyPI)"
	@echo \#
	@echo "#	make pep8			Check for PEP8 compliance"
	@echo "#	make test			Run fast and slow test suites"
	@echo "#	make test-fast			Run fast test suite"
	@echo "#	make test-slow			Run slow test suite"
	@echo "#	make kernel			Setup Jupyter for tests"
	@echo "#	make test-jupyter		Test the Jupyter examples"
	@echo \#
	@echo \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
	@echo


venv:
	python3 -m venv ./venv
	@echo
	@echo Built virtual environment in ./venv
	@echo Run \'source venv/bin/activate\' to activate it!

.PHONY: doc setup test test-fast test-slow kernel test-jupyter clean

doc:
        # The apidoc call is long because we need to tell it to
        # use the venv's version of sphinx-build
	sphinx-apidoc -f -o docs/source/ leap/ SPHINXBUILD='python $(shell which sphinx-build)'
	cd docs && make html

setup:
	pip install -r requirements_freeze.txt
	python setup.py develop

dist:
	pip install setuptools wheel
	python setup.py sdist bdist_wheel

test:
	# Default options are configured in pytest.ini
	# Skip jupyter tests, because they only work if the kernel is configured manually
	py.test -m "not jupyter"

test-fast:
	py.test -m "not system and not jupyter"

test-slow:
	py.test -m system

kernel:
	# Setup a kernel for Jupyter with the name test-jupyter uses to find it
	python -m ipykernel install --user --name="LEAP_venv"

test-jupyter:
	# Won't work unless you have a 'LEAP_venv' kernel
	py.test -m jupyter

pep8:
	# Check for PEP8 compliance in source directories
	flake8 leap examples

clean:
	cd docs && make clean
