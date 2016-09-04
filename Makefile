OPTION = null

all: build test

build: srilm/*.pyx srilm/*.pxd
	python3 setup.py build_ext --inplace --srilm-option $(OPTION)

egg: build
	python3 setup.py bdist_egg

docs:
	sphinx-apidoc -f -o doc/ srilm/
	cd doc; make html; cd -

test:
	python3 -m unittest discover -v tests/

check-style:
	flake8 --show-source --ignore=E501 srilm/*.py tests/*.py *.py
	flake8 --show-source --ignore=E501,E225,E901,E402 srilm/*.pyx

clean:
	python3 setup.py clean
	-rm -rf srilm/*.cpp srilm/*.so srilm/*.html build dist srilm.egg-info
