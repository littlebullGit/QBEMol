"""Compatibility subset retained for molecular QBEMol imports.

The public periodic-BE entry points from upstream QuEmb are intentionally not
part of this paper release. Molecular BE modules still import ``Frags`` from
``quemb.kbe.pfrag`` for shared optimizer/fragment compatibility, so that small
subset remains available.
"""

from quemb.kbe.pfrag import Frags

__all__ = ["Frags"]
