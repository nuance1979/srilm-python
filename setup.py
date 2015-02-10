from distutils.core import setup, Extension
from Cython.Build import cythonize

machine_type = 'i686-m64'

module_dict = {
    'vocab' : 'srilm/vocab.pyx',
    'ngram' : 'srilm/ngram.pyx',
    }

modules = []
for n, s in module_dict.iteritems():
    modules.append(Extension(
        name = n,
        sources = [s],
        language = 'c++',
        define_macros = [('HAVE_ZOPEN','1')],
        include_dirs = ['../include'],
        libraries = ['lbfgs'],
        library_dirs = ['../lib/%s' % machine_type, '/lm/scratch/yi_su/local/lib'],
        extra_compile_args = ['-fopenmp'],
        extra_link_args = ['-fopenmp'],
        extra_objects = ['../lib/%s/liboolm.a' % machine_type, 
                         '../lib/%s/libdstruct.a' % machine_type,
                         '../lib/%s/libmisc.a' % machine_type,
                         '../lib/%s/libz.a' % machine_type]
        ))


setup(
    name = 'srilm',
    version = '0.1',
    description = 'Python binding for SRI Language Modeling Toolkit implemented in Cython',
    author = 'Yi Su',
    author_email = 'nuance1979@hotmail.com',
    packages = ['srilm'],
    ext_package = 'srilm',
    ext_modules = cythonize(modules)
)
