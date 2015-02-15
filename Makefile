all: build

build:
	python2.7 setup.py build_ext --inplace

doc: doc/index.rst doc/conf.py
	cd doc; make html; cd -

test:
	python2.7 -m unittest discover tests/

clean:
	python2.7 setup.py clean
