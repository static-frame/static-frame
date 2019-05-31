# include "array_go.h"


static int update_array_cache(ArrayGOObject* self) {

    PyObject* container;
    PyObject* temp;

    if (self->list) {

        if (self->array) {

            container = PyTuple_Pack(2, (PyObject*) self->array, (PyObject*) self->list);

            if (!container) {
                return -1;
            }

            temp = (PyObject*) self->array;
            self->array = (PyArrayObject*) PyArray_Concatenate(container, 0);
            Py_DECREF(container);
            Py_DECREF(temp);

        } else {

            self->array = (PyArrayObject*) PyArray_FROM_OT((PyObject*) self->list, self->dtype->type_num);
        }

        PyArray_CLEARFLAGS(self->array, NPY_ARRAY_WRITEABLE);

        temp = (PyObject*) self->list;
        self->list = NULL;
        Py_DECREF(temp);
    }

    return 0;
}

/* Methods: */


static int ArrayGO___init__(ArrayGOObject* self, PyObject* args, PyObject* kwargs) {

    PyObject* temp;
    PyObject* iterable;
    int own_iterable;
    int parsed;

    char* argnames[] = {"iterable", "dtype", "own_iterable", NULL};

    temp = (PyObject*) self->dtype;
    self->dtype = NULL;
    Py_XDECREF(temp);

    parsed = PyArg_ParseTupleAndKeywords(
        args, kwargs, "O|$O&p:ArrayGO", argnames,
        &iterable, PyArray_DescrConverter, &self->dtype, &own_iterable
    );

    if (!parsed) {
        return -1;
    }

    if (!self->dtype) {
        self->dtype = PyArray_DescrFromType(NPY_OBJECT);
    }

    if (PyArray_Check(iterable)) {

        temp = (PyObject*) self->array;

        if (own_iterable) {
            PyArray_CLEARFLAGS((PyArrayObject*) iterable, NPY_ARRAY_WRITEABLE);
            Py_INCREF(iterable);
        } else {
            iterable = (PyObject*) SFUtil_ImmutableFilter((PyArrayObject*) iterable);
        }

        if (!PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*) iterable), self->dtype)) {
            PyErr_Format(
                PyExc_TypeError, "bad dtype given to ArrayGO initializer (expected '%S', got '%S')",
                PyArray_DESCR((PyArrayObject*) iterable), self->dtype
            );
            return -1;
        }

        self->array = (PyArrayObject*) iterable;
        Py_XDECREF(temp);

        temp = (PyObject*) self->list;
        self->list = NULL;
        Py_XDECREF(temp);

    } else {

        temp = (PyObject*) self->list;

        if (PyList_Check(iterable) && own_iterable) {
            Py_INCREF(iterable);
        } else {
            iterable = PySequence_List(iterable);
        }

        self->list = (PyListObject*) iterable;
        Py_XDECREF(temp);

        temp = (PyObject*) self->array;
        self->array = NULL;
        Py_XDECREF(temp);
    }

    return 0;
}


static PyObject* ArrayGO_append(ArrayGOObject* self, PyObject* value) {

    if (!self->list) {

        self->list = (PyListObject*) PyList_New(1);

        if (!self->list) {
            return NULL;
        }

        Py_INCREF(value);
        PyList_SET_ITEM(self->list, 0, value);

    } else if (PyList_Append((PyObject*) self->list, value)) {

        return NULL;
    }

    Py_RETURN_NONE;
}


static PyObject* ArrayGO_extend(ArrayGOObject* self, PyObject* values) {

    Py_ssize_t len;

    if (!self->list) {

        self->list = (PyListObject*) PySequence_List(values);

        if (!self->list) {
            return NULL;
        }

    } else {

        len = PyList_Size((PyObject*) self->list);

        if (len < 0 || PyList_SetSlice((PyObject*) self->list, len, len, values)) {
            return NULL;
        }
    }

    Py_RETURN_NONE;
}


static PyObject* ArrayGO_copy(ArrayGOObject* self, PyObject* Py_UNUSED(unused)) {

    ArrayGOObject* copy;

    copy = PyObject_New(ArrayGOObject, &ArrayGOType);

    copy->array = self->array;
    copy->list = (PyListObject*) PySequence_List((PyObject*) self->list);
    copy->dtype = self->dtype;

    Py_XINCREF(copy->array);
    Py_INCREF(copy->dtype);

    return (PyObject*) copy;
}


static PyObject* ArrayGO___iter__(ArrayGOObject* self){
    return (self->list && update_array_cache(self)) ? NULL : PyObject_GetIter((PyObject*) self->array);
}


static PyObject* ArrayGO___getitem__(ArrayGOObject* self, PyObject* key) {
    return (self->list && update_array_cache(self)) ? NULL : PyObject_GetItem((PyObject*) self->array, key);
}


static Py_ssize_t ArrayGO___len__(ArrayGOObject* self) {
    return (self->array ? PyArray_SIZE(self->array) : 0) + (self->list ? PyList_Size((PyObject*) self->list) : 0);
}


static PyObject* ArrayGO_values___get__(ArrayGOObject* self, void* Py_UNUSED(closure)) {
    return (self->list && update_array_cache(self)) ? NULL : (Py_INCREF(self->array), (PyObject*) self->array);
}


/* Not really __del__ (a finalizer), but actually a deallocator: */
/* Not accessible from the Python layer! */


static void ArrayGO___del__(ArrayGOObject* self) {

    Py_XDECREF(self->dtype);
    Py_XDECREF(self->array);
    Py_XDECREF(self->list);

    Py_TYPE(self)->tp_free((PyObject*) self);
}


/* Bindings: */


static struct PyGetSetDef ArrayGO_properties[] = {

    /* {name, getter, setter, docstring, closure} */

    {"values", (getter)ArrayGO_values___get__, NULL, ArrayGO_values___doc__, NULL},

    {NULL},
};


static PyMethodDef ArrayGO_methods[] = {

    /* {name, function, flags, docstring} */

    {"append", (PyCFunction) ArrayGO_append, METH_O, NULL},
    {"extend", (PyCFunction) ArrayGO_extend, METH_O, NULL},
    {"copy", (PyCFunction) ArrayGO_copy, METH_NOARGS, ArrayGO_copy___doc__},

    {NULL},
};


static PyMappingMethods ArrayGO_as_mapping = {
    (lenfunc) ArrayGO___len__,        /* mp_len */
    (binaryfunc) ArrayGO___getitem__, /* mp_subscript */
    0,                                /* mp_ass_subscript */
};


static PyTypeObject ArrayGOType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ArrayGO",                      /* tp_name */
    sizeof(ArrayGOObject),          /* tp_basicsize */
    0,                              /* tp_itemsize */
    (destructor) ArrayGO___del__,   /* tp_dealloc */
    0,                              /* tp_print */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_reserved */
    0,                              /* tp_repr */
    0,                              /* tp_as_number */
    0,                              /* tp_as_sequence */
    &ArrayGO_as_mapping,            /* tp_as_mapping */
    0,                              /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,             /* tp_flags */
    ArrayGO___doc__,                /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    (getiterfunc) ArrayGO___iter__, /* tp_iter */
    0,                              /* tp_iternext */
    ArrayGO_methods,                /* tp_methods */
    0,                              /* tp_members */
    ArrayGO_properties,             /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc) ArrayGO___init__,    /* tp_init */
    0,                              /* tp_alloc */
    PyType_GenericNew,              /* tp_new */
};


/* Boilerplate: */


static struct PyModuleDef array_go_module = {
    PyModuleDef_HEAD_INIT, "array_go", NULL, -1, NULL,
};


PyObject* PyInit_array_go(void) {

    PyObject* array_go;

    import_array();

    if (
        !(array_go = PyModule_Create(&array_go_module))
        || PyType_Ready(&ArrayGOType)
        || PyModule_AddObject(array_go, "ArrayGO", (PyObject*) &ArrayGOType)
    ) {
        return NULL;
    }

    return array_go;
}
