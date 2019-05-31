# ifndef ARRAY_GO_H
# define ARRAY_GO_H

# include "SF.h"


typedef struct {
    PyObject_VAR_HEAD
    PyArrayObject* array;
    PyListObject* list;
    PyArray_Descr* dtype;
} ArrayGOObject;


static PyTypeObject ArrayGOType;


PyDoc_STRVAR(
    ArrayGO___doc__,
    "\n"
    "A grow only, one-dimensional, object type array, "
    "specifically for usage in IndexHierarchy IndexLevel objects.\n"
    "\n"
    "Args:\n"
    "    own_iterable: flag iterable as ownable by this instance.\n"
);


PyDoc_STRVAR(
    ArrayGO_copy___doc__,
    "Return a new ArrayGO with an immutable array from this ArrayGO\n"
);


PyDoc_STRVAR(ArrayGO_values___doc__, "Return the immutable labels array\n");

# endif
