setup:
	python setup.py develop

.PHONY: test
test:
	py.test --doctest-modules


