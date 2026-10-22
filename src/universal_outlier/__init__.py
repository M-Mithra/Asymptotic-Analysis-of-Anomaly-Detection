"""Universal outlier hypothesis testing: fixed-sample and sequential tests.

M-Tech project (MTP) code base. The package is organised as

* ``divergences``  - KL, Bhattacharyya, TVD, Chernoff for discrete pmfs
* ``mmd``          - kernel / MMD estimators (from samples and from counts)
* ``detectors``    - universal outlier decision rules (KL, MMD, GLT, ...)
* ``fixed_sample`` - Monte-Carlo and exact error probabilities P_e(n)
* ``exponents``    - estimating the error exponent c from P_e(n)
* ``theory``       - Sanov (large-deviation) error exponents of each rule
* ``regions``      - decision regions in type space (the "cube" plots)
* ``sequential``   - the NP-SEQ sequential MMD test
* ``alpha``        - threshold decay exponent study (MTP-2)
"""

__version__ = "1.0.0"
