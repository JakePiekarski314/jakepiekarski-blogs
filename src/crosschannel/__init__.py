"""Cross-channel interaction MMM: data, model, and render-safe reporting helpers.

The ``report`` module is deliberately free of PyMC / pymc-marketing imports so it
can run at Quarto render time using only numpy / pandas / matplotlib / xarray.
The ``model`` module is only used offline by ``scripts/fit_cross_channel.py``.
"""
