# include "util.h"


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

    if (PyArray_EquivTypes(d1, d2)) {
        Py_INCREF(d1);
        return d1;
    }

    if (
        PyDataType_ISOBJECT(d1) || PyDataType_ISOBJECT(d2)
        || PyDataType_ISBOOL(d1) || PyDataType_ISBOOL(d2)
        || (PyDataType_ISSTRING(d1) != PyDataType_ISSTRING(d2))
    ) {
        return PyArray_DescrFromType(NPY_OBJECT);
    }

    return PyArray_PromoteTypes(d1, d2);
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


static PyObject* immutable_filter(PyObject* Py_UNUSED(util), PyObject* arg) {

    if (!PyArray_Check(arg)) {
        return PyErr_Format(
            PyExc_TypeError, "_immutable_filter() argument 2 must be %s, not %s",
            ((PyTypeObject*) &PyArray_Type)->tp_name, Py_TYPE(arg)->tp_name
        );
    }

    return (PyObject*) SFUtil_ImmutableFilter((PyArrayObject*) arg);
}


static PyObject* mloc(PyObject* Py_UNUSED(util), PyObject* arg) {

    if (!PyArray_Check(arg)) {
        return PyErr_Format(
            PyExc_TypeError, "mloc() argument 2 must be %s, not %s",
            ((PyTypeObject*) &PyArray_Type)->tp_name, Py_TYPE(arg)->tp_name
        );
    }

    return PyLong_FromVoidPtr(PyArray_DATA((PyArrayObject*) arg));
}


static PyObject* resolve_dtype_iter(PyObject* Py_UNUSED(util), PyObject* arg) {
    return (PyObject*) SFUtil_ResolveDTypesIter(arg);
}


static PyObject* _resolve_dtype(PyObject* Py_UNUSED(util), PyObject* args) {

    PyArray_Descr *d1;
    PyArray_Descr *d2;

    if (
        !PyArg_ParseTuple(
            args, "O!O!:_resolve_dtype",
            &PyArrayDescr_Type, &d1, &PyArrayDescr_Type, &d2
        )
    ) {
        return NULL;
    }

    return (PyObject*) SFUtil_ResolveDTypes(d1, d2);
}


static PyObject* name_filter(PyObject* Py_UNUSED(util), PyObject* arg) {

    if (PyObject_Hash(arg) == -1) {
        return PyErr_Format(
            PyExc_TypeError, "Unhashable name (type '%s').",
            Py_TYPE(arg)->tp_name
        );
    }

    Py_INCREF(arg);
    return arg;
}


static PyMethodDef UtilMethods[] =  {

    {"immutable_filter", immutable_filter, METH_O, NULL},
    {"mloc", mloc, METH_O, NULL},
    {"name_filter", name_filter, METH_O, NULL},
    {"resolve_dtype_iter", resolve_dtype_iter, METH_O, NULL},
    {"_resolve_dtype", _resolve_dtype, METH_VARARGS, NULL},

    {NULL, NULL, 0, NULL},

};


static struct PyModuleDef Util = {
    PyModuleDef_HEAD_INIT, "util", NULL, -1, UtilMethods,
};


PyObject* PyInit_util(void) {

    import_array();

    return PyModule_Create(&Util);
}
