venv:
	python3 -m venv ./venv
	@echo
	@echo Built virtual environment in ./venv
	@echo Run \'source venv/bin/activae\' to activate it!

.PHONY: setup test doc

doc:
	sphinx-apidoc -o docs/source/ src/ SPHINXBUILD='python $(shell which sphinx-build)'
	cd docs && make html

setup:
	pip install -r requirements.txt
	python setup.py develop

test:
	py.test --doctest-modules


