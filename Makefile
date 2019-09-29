help:
	@echo \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
	@echo \# LEAP Makefile
	@echo \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
	@echo \#
	@echo \# First build a virtual environment:
	@echo \#
	@echo \#\	make venv
	@echo \#
	@echo \# Then activate it:
	@echo \#
	@echo \#\	source venv/bin/activate
	@echo \#
	@echo \# Then setup the environment:
	@echo \#
	@echo \#\	make setup
	@echo \#
	@echo \# And run tests and build docs:
	@echo \#
	@echo \#\	make test
	@echo \#\	make doc
	@echo \#
	@echo \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
	@echo


venv:
	python3 -m venv ./venv
	@echo
	@echo Built virtual environment in ./venv
	@echo Run \'source venv/bin/activate\' to activate it!

.PHONY: setup test doc

doc:
        # The apidoc call is long because we need to tell it to
        # use the venv's version of sphinx-build
	sphinx-apidoc -f -o docs/source/ src/ SPHINXBUILD='python $(shell which sphinx-build)'
	cd docs && make html

setup:
	pip install -r requirements.txt
	python setup.py develop

test:
	py.test --doctest-modules --cov=src/ --cov-report=html
