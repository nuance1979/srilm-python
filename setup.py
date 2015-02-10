from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(
    name = 'srilm',
    version = '0.1',
    description = 'Python binding for SRI Language Modeling Toolkit implemented in Cython',
    author = 'Yi Su',
    author_email = 'nuance1979@hotmail.com',
    packages = ['srilm'],
    ext_package = 'srilm',
    ext_modules = cythonize(Extension(
            name = 'vocab',
            sources = ['srilm/vocab.pyx'],
            language = 'c++',
            define_macros = [('HAVE_ZOPEN','1')],
            include_dirs = ['../include'],
            library_dirs = ['../lib/macosx'],
            extra_objects = ["../lib/macosx/liboolm.a", 
                             "../lib/macosx/libdstruct.a",
                             "../lib/macosx/libmisc.a",
                             "../lib/macosx/libz.a"],
            )))

