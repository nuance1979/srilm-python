from distutils.core import setup, Extension
from Cython.Build import cythonize
import subprocess

machine_type = subprocess.check_output(["/bin/bash", "../sbin/machine-type"]).strip()

if machine_type == 'i686-m64':
    compile_args = ['-fopenmp']
    link_args = ['-fopenmp']
    lib_dirs = ['/lm/scratch/yi_su/local/lib']
elif machine_type == 'macosx':
    compile_args = None
    link_args = None
    lib_dirs = ['/opt/local/lib']

module_dict = {
    'vocab' : 'srilm/vocab.pyx',
    'stats' : 'srilm/stats.pyx',
    'discount' : 'srilm/discount.pyx',
    'base' : 'srilm/base.pyx',
    'ngram' : 'srilm/ngram.pyx',
    'maxent' : 'srilm/maxent.pyx',
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
        library_dirs = ['../lib/%s' % machine_type] + lib_dirs,
        extra_compile_args = compile_args,
        extra_link_args = link_args,
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
    license = 'MIT',
    packages = ['srilm'],
    ext_package = 'srilm',
    ext_modules = cythonize(modules, annotate=True)
)
