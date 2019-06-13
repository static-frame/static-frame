# ifndef SF_H
# define SF_H

# include "Python.h"
# include "structmember.h"

# define PY_ARRAY_UNIQUE_SYMBOL SF_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"


/* Bug in NumPy < 1.16: https://github.com/numpy/numpy/pull/12131 */

# undef PyDataType_ISBOOL
# define PyDataType_ISBOOL(obj) PyTypeNum_ISBOOL(((PyArray_Descr*)(obj))->type_num)


/* SF APIs: */

PyArrayObject* SFUtil_ImmutableFilter(PyArrayObject*);

PyArray_Descr* SFUtil_ResolveDTypes(PyArray_Descr*, PyArray_Descr*);

PyArray_Descr* SFUtil_ResolveDTypesIter(PyObject*);

# endif
