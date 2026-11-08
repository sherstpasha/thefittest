Multiobjective utilities
========================

Sorting and dominance comparisons expect maximization fitness. Negate columns
that should be minimized before calling them directly. Crowding distance accepts
either objective direction but must be evaluated within one front.

The convergence metric measures obtained-to-reference distances (not IGD).
The diversity metric measures two-objective spread for minimization fronts.
Neither metric normalizes objective units.

.. automodule:: thefittest.utils.multiobjective
   :members:
   :undoc-members:
