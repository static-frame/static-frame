



Title:

Improve Code Quality with Array and DataFrame Type Hints
Type-Hinting Arrays and DataFrames for Static Analysis and Runtime Validation

How Type Annotations of Arrays and DataFrames Improve Code
Type-Hinting Generic Arrays and DataFrames
How Type-Hinting Arrays and DataFrames Improve Your Code


// Abstract
// Your Abstract will appear in the online schedule and give attendees a sense of your talk. This should be around 100 words or less.


As tools for Python type annotations have evolved, more complex data structures can be typed, providing improved readability and static analysis. Arrays and DataFrames, as complex containers, have only recently supported complete type annotations in Python. NumPy 1.20 introduced generic specification of arrays and dtypes. Building on NumPy's foundation, StaticFrame 2.0 introduced complete type specification of DataFrames, employing NumPy primitives and variadic generics.

This talk will introduce practical approaches to type-hinting arrays and DataFrames, and show how they improve code quality with static analysis and run-time validation. How these libraries specify type requirements, along with the usage of modern typing tools, will also be demonstrated.





// Description
// Your placement in the program will be based on reviews of your description. This should be a roughly 500-word outline of your presentation. This outline should concisely describe software of interest to the SciPy community, tools or techniques for more effective computing, or how scientific Python was applied to solve a research problem. A traditional background/motivation, methods, results, and conclusion structure is encouraged but not required. Links to project websites, source code repositories, figures, full papers, and evidence of public speaking ability are encouraged.


As the tools for Python type annotations have improved, more complex data structures can be typed. Using TypeHints in Python provides documentation, supports static type-checking, facilitates integration with integrated development environments, and can be used for run-time validation.

Arrays and DataFrames, as complex containers, have only recently supported complete type annotations in Python. NumPy 1.20 introduced generic specification of arrays and dtypes. Building on NumPy's foundation, StaticFrame 2.0 introduced complete type specification of DataFrames, employing NumPy primitives and variadic generics. With these tools, DataFrames with heterogeneously typed and variably sized columns can be typed.

This talk will introduce practical approaches to type-hinting arrays and DataFrames, and demonstrate their application for static analysis and run-time validation. As complete type specification can be daunting, opportunities for incrementally integrating type hints will be demonstrated. Usage of ``mypy`` and ``pyright`` for static analysis, and StaticFrame's ``TypeClinic`` and ``CallGuard`` for run-time validation, will be shown.

Further, how these libraries specify type requirements, along with the usage of modern typing tools such as `overload`, `Literal`, `TypeVar`, and `TypeVarTuple`, will also be demonstrated. For those seeking to add type hints to their libraries, these examples will demonstrate the range of typing flexibility now possible in Python.


The NumPy interface module:
https://github.com/numpy/numpy/blob/main/numpy/__init__.pyi

Introduction to typing in StaticFrame:
https://static-frame.readthedocs.io/en/latest/articles/ftyping.html

PEP 646 on Variadic Generics
https://peps.python.org/pep-0646/

Example of a Part Presentation (SciPy 2023)
https://www.youtube.com/watch?v=i4IqWD1zBuo

Example of a Past Presentation (PyCon 2023)
https://www.youtube.com/watch?v=ppPXPVV4rDc



For one example, by defining overloaded function definitions using `Literal` arguments to encode integer `axis` values, axis-specific return types can be specified. For another example, `TypeVarTuple` permits defining contiguous regions of uniform types, permitting expressive specifcation of a wide range of types.



