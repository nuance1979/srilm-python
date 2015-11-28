OPTION = null

all: build test

build: srilm/*.pyx srilm/*.pxd
	python2.7 setup.py build_ext --inplace --srilm-option $(OPTION)

docs:
	sphinx-apidoc -f -o doc/ srilm/
	cd doc; make html; cd -

test:
	python2.7 -m unittest discover -v tests/

clean:
	python2.7 setup.py clean
	-rm srilm/*.cpp srilm/*.so srilm/*.html
