# Data provenance

## eyam_1666.csv

Susceptible and infective counts for the plague outbreak in the village of Eyam, Derbyshire,
England, between 18 June and 20 October 1666. The village isolated itself during the outbreak,
which makes it a standard small, closed-population data set for SIR fitting.

**Primary source.**

> Raggett, G. (1982). A stochastic model of the Eyam plague. *Journal of Applied Statistics*,
> 9(2), 212–225.

Raggett obtained the counts from the parish list of deaths. The infective population is
estimated from the list of future deaths by assuming a fixed length of illness before death,
and the susceptible population follows because the village was isolated.

**Digitization used here.** The values are taken from Raggett's table as reproduced, identically,
in two independent publications:

- L. S. T. Ho, J. Xu, F. W. Crawford, V. N. Minin and M. A. Suchard, *Birth/birth-death
  processes and their computable transition probabilities with biological applications*,
  arXiv:1603.03819, Table 1;
- A. Golightly and C. Sherlock, *Efficient sampling of conditioned Markov jump processes*,
  arXiv:1809.07139, Table 4.

The same table ships as the `Eyam` data set of the R package `MultiBD`.

| Column | Meaning |
|---|---|
| `month` | time in months after 18 June 1666, as published (0, 0.5, …, 3, 4) |
| `day` | the same time in days, `month × 365.25 / 12` (rates in this project are per day) |
| `susceptible` | susceptible count S |
| `infective` | estimated infective count I |

The model uses N = S + I at t = 0 = 254 + 7 = 261 and I₀ = 7 (`configs/e14.yaml`). The fit in
`experiments/E14_realdata.py` uses the `day` and `susceptible` columns.

**History.** An earlier version of this file had the commonly reproduced susceptible counts but an
invented time axis: eight points 14 days apart, ending on 25 September. It was replaced by the
verified table above, with the real observation times.
