# Lethola 2019

## Guess Assesment
Lethola uses the projection of the guess orbitals onto the converged SCF WF to determine the quality of the initial guess. 

-> Metric fraction of e- density covered by the guess orbitals.

Pro: 
- Relatively simple and yield fine grained continuous information for overlap instead of discreate value (like iterations count). 
- Only a single SCF calculation needed. 
- 
Con: 
- No direct information about convergence speed, only under the assumption that a close guess converges faster.

## Open problems
Lethola used QChem for his SCF calcualtions. However, there still exists the problem that guesses may yield greatly different results in different programs.
