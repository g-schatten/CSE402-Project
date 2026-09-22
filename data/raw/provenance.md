# Data provenance

## eyam_1666.csv

The 1665-1666 plague outbreak in the village of Eyam, Derbyshire, England, is
the standard "hello world" dataset for SIR model fitting (small, closed
population, famously well-documented because the village self-quarantined).

The commonly cited source for the (day, S) counts used in numerical-methods
courses is:

> Raggett, G. F. (1982). "Modeling the Eyam plague." *Bulletin of the
> Institute of Mathematics and its Applications*, 18, 221-226.

with often-repeated total population N = 261 and initial infectious I0 = 7
(so S0 = N - I0 = 254 at t=0).

**Action item before the report/deck cite these numbers:** the digitized
(day, S) pairs in `eyam_1666.csv` were reconstructed from memory of the
widely-reproduced version of this table (as it appears in numerous SIR
teaching materials) and have **not** been verified against Raggett (1982)
directly in this session (no network access was used to fetch the primary
source). Before E14's fit is reported as a real result:

1. Locate a verified digitization of Raggett's table (library copy, a
   textbook that reproduces it with citation, or a maintained dataset
   repository) and replace `eyam_1666.csv` if the numbers differ.
2. If the values already match, remove this caveat and record the exact
   source (page/table number) here instead.

This does not block development: the numerical pipeline (E14) is written
against the CSV's schema (`day,date,susceptible`) regardless of the exact
values, so swapping in verified numbers later requires no code changes.
