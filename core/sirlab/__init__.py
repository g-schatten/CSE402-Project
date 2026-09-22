"""SIRLab: a hand-written numerical library for the SIR family of epidemic models.

This package implements every numerical method it uses from scratch (solvers,
linear algebra, quadrature, root-finding, optimization, and MCMC). SciPy is
used only inside the test suite, to cross-check these implementations against
a trusted reference -- never inside sirlab itself. See PLAN.md section 3 for
the architectural contract.
"""

__version__ = "0.1.0"
