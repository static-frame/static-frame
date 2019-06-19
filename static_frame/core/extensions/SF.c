# define NO_IMPORT_ARRAY
# include "SF.h"


PyArrayObject* SFUtil_ImmutableFilter(PyArrayObject* src_array) {

    PyArrayObject* dst_array;

    if (PyArray_FLAGS(src_array) & NPY_ARRAY_WRITEABLE) {
        dst_array = (PyArrayObject*) PyArray_NewCopy(src_array, NPY_ANYORDER);
        PyArray_CLEARFLAGS(dst_array, NPY_ARRAY_WRITEABLE);
        return dst_array;
    }

    Py_INCREF(src_array);
    return src_array;
}


PyArray_Descr* SFUtil_ResolveDTypes(PyArray_Descr* d1, PyArray_Descr* d2) {

    PyArray_Descr* result;

    if (PyArray_EquivTypes(d1, d2)) {
        Py_INCREF(d1);
        return d1;
    }

    if (
        PyDataType_ISOBJECT(d1)
        || PyDataType_ISOBJECT(d2)
        || PyDataType_ISBOOL(d1)
        || PyDataType_ISBOOL(d2)
        || (PyDataType_ISSTRING(d1) != PyDataType_ISSTRING(d2))
        || (
            /* PyDataType_ISDATETIME matches both NPY_DATETIME *and* NPY_TIMEDELTA,
            So we need the PyArray_EquivTypenums check too: */
            (PyDataType_ISDATETIME(d1) || PyDataType_ISDATETIME(d2))
            && !PyArray_EquivTypenums(d1->type_num, d2->type_num)
        )
    ) {
        return PyArray_DescrFromType(NPY_OBJECT);
    }

    result = PyArray_PromoteTypes(d1, d2);

    if (!result) {
        PyErr_Clear();
        return PyArray_DescrFromType(NPY_OBJECT);
    }

    return result;
}


PyArray_Descr* SFUtil_ResolveDTypesIter(PyObject* dtypes) {

    PyObject* iterator;
    PyArray_Descr* resolved;
    PyArray_Descr* dtype;
    PyArray_Descr* temp;

    iterator = PyObject_GetIter(dtypes);

    if (iterator == NULL) {
        return NULL;
    }

    resolved = NULL;

    while ((dtype = (PyArray_Descr*) PyIter_Next(iterator))) {

        if (!PyArray_DescrCheck(dtype)) {

            PyErr_Format(
                PyExc_TypeError, "argument must be an iterable over %s, not %s",
                ((PyTypeObject*) &PyArrayDescr_Type)->tp_name, Py_TYPE(dtype)->tp_name
            );

            Py_DECREF(iterator);
            Py_DECREF(dtype);
            Py_XDECREF(resolved);

            return NULL;
        }

        if (!resolved) {
            resolved = dtype;
            continue;
        }

        temp = SFUtil_ResolveDTypes(resolved, dtype);

        Py_DECREF(resolved);
        Py_DECREF(dtype);

        resolved = temp;

        if (!resolved || PyDataType_ISOBJECT(resolved)) {
            break;
        }
    }

    Py_DECREF(iterator);

    return resolved;
}
