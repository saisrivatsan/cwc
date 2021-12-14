#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils import sysconfig
import glob

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# gather up all the source files
srcFiles = ['mithral.i']
includeDirs = [numpy_include]
srcDir = os.path.abspath('src')
for root, dirnames, filenames in os.walk(srcDir):
  for dirname in dirnames:
    absPath = os.path.join(root, dirname)
    #print('adding dir to path: %s' % absPath)
    globStr = "%s/*.c*" % absPath
    files = glob.glob(globStr)
    #print(files)
    includeDirs.append(absPath)
    srcFiles += files

print("includeDirs:")
print(includeDirs)
print("srcFiles:")
print(srcFiles)


os.environ["CC"] = "g++" # force compiling c as c++
extra_args = ['-std=c++14','-fno-rtti','-march=native', '-ffast-math']

# inplace extension module
_mithral = Extension("_mithral",
                   srcFiles,
                   #define_macros=[('NDEBUG', '1')],
                   include_dirs=includeDirs,
                   swig_opts=['-c++'],
                   extra_compile_args=extra_args,
                   )

# NumyTypemapTests setup
setup(  name        = "Mithral Wrapper",
        description = "Wrap a C/C++ Mithral library in Numpy",
        author      = "Sai Srivatsa Ravindranath",
        version     = "1.0",
        license     = "MIT",
        ext_modules = [_mithral]
        )



