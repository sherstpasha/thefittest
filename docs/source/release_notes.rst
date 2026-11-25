Release notes
=============

0.3.0
-----

- Updated SHAGA with configurable selection, crossover and multiparent operators.
- Added SelfCSHAGA with self-configuring selection and crossover probabilities.
- Added PDPSHAGA with operator probabilities adapted from strict fitness improvements.
- Added histories of mutation/crossover rate memories and operator probabilities.
- Added NSGAII and NSGPII with their SelfC and PDP adaptive variants.
- Added Pareto-front retrieval, nondominated sorting and crowding-distance utilities.
- Added rank/crowding and dominance/crowding tournament selection.
- Added convergence and spread metrics.
- Added SCH, FON, POL, KUR and ZDT1/2/3/4/6 benchmark problems.
- Added vector fitness caching, front-stability stopping and per-objective targets.
- Added prediction-loss/model-complexity objectives to symbolic regression and GPNN estimators.
- Added Pareto-model selection and switching without refitting.
- Enabled growing mutation to regrow single-terminal trees within the depth limit.
- Corrected ``logabs`` and enabled ``sqrtabs``; re-evaluate experiments using the old ``logabs`` mapping.
- Corrected the regressor export list to include ``GeneticProgrammingRegressor``.
- Corrected the description of ``ln`` activation to linear identity.
- Set Python compatibility to 3.9–3.13, including patch releases.
- Set minimum dependencies to NumPy 1.26 and scikit-learn 1.6.
- Added the ``torch`` installation extra with CPU and optional CUDA support.
- Archived documentation 0.2.7 and added version navigation.
- Added Learning Materials with repository-owned notebooks for 0.3.0.
