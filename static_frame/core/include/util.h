# ifndef UTIL_H
# define UTIL_H

# include "Python.h"

# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"


/* Until we require Numpy 1.16.0 or above: */

# undef PyDataType_ISBOOL
# define PyDataType_ISBOOL(obj) PyTypeNum_ISBOOL(((PyArray_Descr*)(obj))->type_num)


PyArrayObject* SFUtil_ImmutableFilter(PyArrayObject*);

PyArray_Descr* SFUtil_ResolveDTypes(PyArray_Descr*, PyArray_Descr*);

PyArray_Descr* SFUtil_ResolveDTypesIter(PyObject*);

# endif
