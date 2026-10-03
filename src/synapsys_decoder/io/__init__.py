"""Dataset loaders. Each loader converts an external format into :class:`Trial` objects.

Loaders never filter or resample; they only reshape, rescale to volts, and
attach provenance.
"""
