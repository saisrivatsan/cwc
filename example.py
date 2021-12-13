# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_example')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_example')
    _example = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_example', [dirname(__file__)])
        except ImportError:
            import _example
            return _example
        try:
            _mod = imp.load_module('_example', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _example = swig_import_helper()
    del swig_import_helper
else:
    import _example
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0

class SimpleStruct(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SimpleStruct, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SimpleStruct, name)
    __repr__ = _swig_repr
    __swig_setmethods__["x"] = _example.SimpleStruct_x_set
    __swig_getmethods__["x"] = _example.SimpleStruct_x_get
    if _newclass:
        x = _swig_property(_example.SimpleStruct_x_get, _example.SimpleStruct_x_set)
    __swig_setmethods__["y"] = _example.SimpleStruct_y_set
    __swig_getmethods__["y"] = _example.SimpleStruct_y_get
    if _newclass:
        y = _swig_property(_example.SimpleStruct_y_get, _example.SimpleStruct_y_set)

    def __init__(self):
        this = _example.new_SimpleStruct()
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _example.delete_SimpleStruct
    __del__ = lambda self: None
SimpleStruct_swigregister = _example.SimpleStruct_swigregister
SimpleStruct_swigregister(SimpleStruct)


def printStuff():
    return _example.printStuff()
printStuff = _example.printStuff

def inplace(inVec):
    return _example.inplace(inVec)
inplace = _example.inplace

def range(outVec, step):
    return _example.range(outVec, step)
range = _example.range

def sumOfArrays(v1, v2):
    return _example.sumOfArrays(v1, v2)
sumOfArrays = _example.sumOfArrays

def timesTwo(v, outVec):
    return _example.timesTwo(v, outVec)
timesTwo = _example.timesTwo

def addArrays(v1, v2, outVec):
    return _example.addArrays(v1, v2, outVec)
addArrays = _example.addArrays

def getIntVect():
    return _example.getIntVect()
getIntVect = _example.getIntVect

def getLongVect():
    return _example.getLongVect()
getLongVect = _example.getLongVect

def getFloatVect():
    return _example.getFloatVect()
getFloatVect = _example.getFloatVect

def getDoubleVect():
    return _example.getDoubleVect()
getDoubleVect = _example.getDoubleVect

def getObjectVect():
    return _example.getObjectVect()
getObjectVect = _example.getObjectVect
class SimpleClass(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SimpleClass, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SimpleClass, name)
    __repr__ = _swig_repr

    def __init__(self, n):
        this = _example.new_SimpleClass(n)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def get_n(self):
        return _example.SimpleClass_get_n(self)
    __swig_destroy__ = _example.delete_SimpleClass
    __del__ = lambda self: None
SimpleClass_swigregister = _example.SimpleClass_swigregister
SimpleClass_swigregister(SimpleClass)

class SimpleArrayClass(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SimpleArrayClass, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SimpleArrayClass, name)
    __repr__ = _swig_repr

    def setArray(self, v):
        return _example.SimpleArrayClass_setArray(self, v)

    def getArray(self, outVec):
        return _example.SimpleArrayClass_getArray(self, outVec)

    def __init__(self):
        this = _example.new_SimpleArrayClass()
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _example.delete_SimpleArrayClass
    __del__ = lambda self: None
SimpleArrayClass_swigregister = _example.SimpleArrayClass_swigregister
SimpleArrayClass_swigregister(SimpleArrayClass)

class ArrayUser_dbl(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ArrayUser_dbl, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ArrayUser_dbl, name)
    __repr__ = _swig_repr

    def __init__(self, scaleBy):
        this = _example.new_ArrayUser_dbl(scaleBy)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _example.delete_ArrayUser_dbl
    __del__ = lambda self: None

    def setArray(self, ar):
        return _example.ArrayUser_dbl_setArray(self, ar)

    def getArray(self, outVec):
        return _example.ArrayUser_dbl_getArray(self, outVec)
ArrayUser_dbl_swigregister = _example.ArrayUser_dbl_swigregister
ArrayUser_dbl_swigregister(ArrayUser_dbl)


def createEigenMat():
    return _example.createEigenMat()
createEigenMat = _example.createEigenMat

def createEigenVect():
    return _example.createEigenVect()
createEigenVect = _example.createEigenVect

def createEigenArray():
    return _example.createEigenArray()
createEigenArray = _example.createEigenArray

def createEigenArrayVect():
    return _example.createEigenArrayVect()
createEigenArrayVect = _example.createEigenArrayVect

def createEigenMatf():
    return _example.createEigenMatf()
createEigenMatf = _example.createEigenMatf

def createEigenVectf():
    return _example.createEigenVectf()
createEigenVectf = _example.createEigenVectf

def createEigenArrayf():
    return _example.createEigenArrayf()
createEigenArrayf = _example.createEigenArrayf

def createEigenArrayVectf():
    return _example.createEigenArrayVectf()
createEigenArrayVectf = _example.createEigenArrayVectf

def createEigenMati():
    return _example.createEigenMati()
createEigenMati = _example.createEigenMati

def createEigenVecti():
    return _example.createEigenVecti()
createEigenVecti = _example.createEigenVecti

def createEigenArrayi():
    return _example.createEigenArrayi()
createEigenArrayi = _example.createEigenArrayi

def createEigenArrayVecti():
    return _example.createEigenArrayVecti()
createEigenArrayVecti = _example.createEigenArrayVecti

def test_code():
    return _example.test_code()
test_code = _example.test_code

def mithral_encode(X, nrows, ncols, splitdims, all_splitvals, scales, offsets, ncodebooks, out):
    return _example.mithral_encode(X, nrows, ncols, splitdims, all_splitvals, scales, offsets, ncodebooks, out)
mithral_encode = _example.mithral_encode

def profile_encode(N, D, nbytes):
    return _example.profile_encode(N, D, nbytes)
profile_encode = _example.profile_encode
# This file is compatible with both classic and new-style classes.


