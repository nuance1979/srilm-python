.. SRILM Python Binding documentation master file, created by
   sphinx-quickstart on Sat Feb 14 17:56:35 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SRILM Python Binding's documentation!
================================================

This project aims to bring the power of the SRILM Toolkit to Python. 

Instead of faithfully wrapping SRILM C++ classes, we create a new set of APIs to give them a Pythonic look-and-feel 
while preserving the raw power of SRILM Toolkit as much as possible. In the process, we also try to 'smooth away' 
some of the idiosyncrasies of the SRILM APIs.

DEPENDENCY
----------

* SRI LM Toolkit >= 1.7.1
* Cython >= 0.20.1
* (optional) Sphinx >= 1.2.2

INSTALL
-------

To get started, first download `SRI Language Modeling Toolkit <http://www.speech.sri.com/projects/srilm>`_ 

Then check out this project and put it *under* the root directory of SRILM::

  $ cd $SRILM
  $ git clone https://github.com/nuance1979/srilm-python

.. note::

   There is a minor bug in SRILM 1.7.1. You can optionally patch it by::

   $ cd srilm-python
   $ patch ../lm/src/MEModel.cc < srilm/MEModel.cc.patch

   Then you need to rebuild SRILM to finish the patch.

Now you can build this project by::

  $ cd srilm-python
  $ make

If successful, you can take a look at the example script::

  $ python ./example.py --help

Or try it interactively by::

  $ python
  ...
  >>> import srilm

DOCUMENTATION
-------------

You can read it here or make it from scratch by::

  $ make docs

UNITTEST
--------

You can run unit tests by::

  $ make test

API
---

.. toctree::
   :maxdepth: 2

   modules.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

