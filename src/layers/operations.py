"""
Provides interface for model operations, arranged in alpha-dimensional order.

If an operation is dimension-specific, then it is specified with an `_Nd` at the end,
where `N` is the amount of dimensions that operation supports.

All operations assume a tensor-shape of [samples, ...] where `...` can be height, width, depth, and
any other combination of dimensions. See `input_` descriptions in docstrings for required shape.
"""
