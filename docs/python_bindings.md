# Python Bindings

The compiled extension is conceptually `neurodic._neurodic`.

```text
bindings/python/module.cpp
bindings/python/bind_*.cpp
```

Bindings expose selected validated C++ interfaces. They must not contain
scientific algorithms and should not automatically expose every internal class.

The thin Python package under `python/neurodic` imports `_neurodic` when it is
available and raises clear errors when the extension has not been built.
