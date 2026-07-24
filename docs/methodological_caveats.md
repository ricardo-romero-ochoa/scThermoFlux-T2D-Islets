# Methodological caveats

## Transcriptomic quasi-potential is not physical free energy

The landscape is defined as:

\[
\Phi(x)=-\log(P(x)+\epsilon)
\]

where \(P(x)\) is the empirical density of cells on a transcriptomic manifold. This is a dimensionless information-theoretic quasi-potential, not a direct measurement of molecular free energy or cellular heat.

## Entropy production is coarse-grained

For a Markov chain over transcriptomic microstates, the entropy-production-like score is:

\[
\sigma = \frac{1}{2}\sum_{i,j} (\pi_iK_{ij}-\pi_jK_{ji})\log\frac{\pi_iK_{ij}+\epsilon}{\pi_jK_{ji}+\epsilon}.
\]

This measures detailed-balance violation in the inferred transcriptomic transition graph. It is not the total thermodynamic entropy production of the cell.

## Cross-sectional data limitations

Most disease scRNA-seq datasets are cross-sectional. The inferred transition graph should be described as disease-associated transition structure, not proven temporal progression.

## Donor-level inference

Cells from the same donor are not independent biological replicates. Statistical tests should be donor-level, or use mixed models if enough metadata exist.

## RNA velocity

This repository does not require RNA velocity for v1. If 10x spliced/unspliced layers are generated, velocity-based transition matrices can be added as optional evidence. Velocity should not be treated as ground truth without robustness checks.
