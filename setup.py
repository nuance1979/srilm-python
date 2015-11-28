from distutils.core import setup, Extension
from Cython.Build import cythonize
import subprocess
import sys

srilm_option = ""
copy_args = sys.argv[1:]
if '--srilm-option' in copy_args:
    ind = copy_args.index('--srilm-option')
    copy_args.pop(ind)
    val = copy_args.pop(ind)
    srilm_option = "" if val == "null" else val

machine_type = subprocess.check_output(["/bin/bash", "../sbin/machine-type"]).strip()
lib_path = machine_type + srilm_option

if machine_type == 'i686-m64':
    compile_args = ['-fopenmp']
    link_args = ['-fopenmp']
    lib_dirs = []
elif machine_type == 'macosx':
    compile_args = None
    link_args = None
    lib_dirs = ['/usr/lib', '/usr/local/lib']

compact_def_macros = [('USE_SARRAY', 1), ('USE_SARRAY_TRIE', 1), ('USE_SARRAY_MAP2', 1)]
if srilm_option == '_c':
    def_macros = compact_def_macros
elif srilm_option == '_s':
    def_macros = compact_def_macros + [('USE_SHORT_VOCAB', 1), ('USE_XCOUNTS', 1)]
elif srilm_option == '_l':
    def_macros = compact_def_macros + [('USE_LONGLONG_COUNTS', 1), ('USE_XCOUNTS', 1)]
else:
    def_macros = []

module_dict = {
    'vocab' : 'srilm/vocab.pyx',
    'stats' : 'srilm/stats.pyx',
    'discount' : 'srilm/discount.pyx',
    'base' : 'srilm/base.pyx',
    'ngram' : 'srilm/ngram.pyx',
    'maxent' : 'srilm/maxent.pyx',
    'utils' : 'srilm/utils.pyx',
    }

modules = []
for n, s in module_dict.iteritems():
    modules.append(Extension(
        name = n,
        sources = [s],
        language = 'c++',
        define_macros = [('HAVE_ZOPEN','1')] + def_macros,
        include_dirs = ['../include'],
        libraries = ['lbfgs'],
        library_dirs = ['../lib/%s' % lib_path] + lib_dirs,
        extra_compile_args = compile_args,
        extra_link_args = link_args,
        extra_objects = ['../lib/%s/liboolm.a' % lib_path,
                         '../lib/%s/libdstruct.a' % lib_path,
                         '../lib/%s/libmisc.a' % lib_path,
                         '../lib/%s/libz.a' % lib_path]
        ))


setup(
    name = 'srilm',
    version = '1.0.0',
    description = 'Python binding for SRI Language Modeling Toolkit implemented in Cython',
    author = 'Yi Su',
    author_email = 'nuance1979@hotmail.com',
    license = 'MIT',
    packages = ['srilm'],
    ext_package = 'srilm',
    ext_modules = cythonize(modules, annotate=True),
    script_args = copy_args
)
