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
	@echo \# Or just run the fast (or slow) test suite:
	@echo \#
	@echo \#\	make test-fast
	@echo \#\	make test-slow
	@echo \#
	@echo \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
	@echo


venv:
	python3 -m venv ./venv
	@echo
	@echo Built virtual environment in ./venv
	@echo Run \'source venv/bin/activate\' to activate it!

.PHONY: setup test doc clean

doc:
        # The apidoc call is long because we need to tell it to
        # use the venv's version of sphinx-build
	sphinx-apidoc -f -o docs/source/ src/ SPHINXBUILD='python $(shell which sphinx-build)'
	cd docs && make html

setup:
	pip install -r requirements.txt
	python -m ipykernel install --user --name="LEAP_venv"
	python setup.py develop

test:
	py.test  # Default options are configured in pytest.ini

test-fast:
	py.test -m "not system"

test-slow:
	py.test -m system

clean:
	cd docs && make clean
