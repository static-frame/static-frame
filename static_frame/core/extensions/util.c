# include "util.h"


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


static PyObject* resolve_dtype(PyObject* Py_UNUSED(util), PyObject* args) {

    PyArray_Descr *d1;
    PyArray_Descr *d2;

    if (
        !PyArg_ParseTuple(
            args, "O!O!:resolve_dtype",
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
    {"resolve_dtype", resolve_dtype, METH_VARARGS, NULL},

    {NULL, NULL, 0, NULL},

};


static struct PyModuleDef Util = {
    PyModuleDef_HEAD_INIT, "util", NULL, -1, UtilMethods,
};


PyObject* PyInit_util(void) {

    import_array();

    return PyModule_Create(&Util);
}
