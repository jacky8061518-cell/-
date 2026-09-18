"""Quantitative research and trading platform.

Four planes, isolated from one another by design rather than by convention:

``data``      acquires, validates and stores. Knows nothing about strategies.
``research``  generates hypotheses, factors and models. Has no broker access.
``trading``   executes approved, versioned models. Cannot invent a model.
``control``   risk, registry, audit, kill switch. Cannot be overridden upstream.

The import direction is enforced by a test: ``research`` may not import
``trading``, and ``control`` may not import either. A research agent that could
reach the execution path would make the whole separation decorative.
"""

__all__ = ["data", "research", "trading", "control"]
