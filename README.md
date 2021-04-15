Metropolis hastings mcmc algorithm

To carry out the Metropolis-Hastings algorithm, we need to draw random samples from the following distributions: 

- the standard uniform distribution 
- a proposal distribution p(x) that we choose to be N(0,σ) 
- the target distribution g(x) which is proportional to the posterior probability

Given an initial guess for θ with positive probability of being drawn, the Metropolis-Hastings algorithm proceeds as follows: 
- Choose a new proposed value (θpp) such that θp=θ+Δθ where Δθ∼N(0,σ) 
- Caluculate the ratio : ρ = g(θp | X)/g(θ | X) where g is the posterior probability.

If the proposal distribution is not symmetrical, we need to weight the acceptance probability to maintain detailed balance (reversibility) of the stationary distribution, and instead calculate: 

ρ=g(θp | X) p(θ | θp) / g(θ | X) p(θp | θ)

Since we are taking ratios, the denominator cancels any distribution proportional to g will also work we can use: 

ρ=p(X|θp)p(θp)/ p(X|θ)p(θ) 

- If ρ≥1, then set θ=θp 
- If ρ<1, then set θ=θp with probability ρ, otherwise set θ=θ (this is where we use the standard uniform distribution)


Repeat the earlier steps. After some number of iterations k, the samples θk+1,θk+2,… will be samples from the posterior distributions. Here are initial concepts to help your intuition about why this is so: 

- We accept a proposed move to θk+1 whenever the density of the (unnormalized) target distribution 
- at θk+1 is larger than the value of θk 
- so θ will more often be found in places where the target distribution is denser If this was all we accepted, θ would get stuck at a local mode of the target distribution, so we also accept occasional moves to lower density regions 
- it turns out that the correct probability of doing so is given by the ratio ρ - The acceptance criteria only looks at ratios of the target distribution, so the denominator cancels out and does not matter 
- that is why we only need samples from a distribution proportional to the posterior distribution So, θ will be expected to bounce around in such a way that its spends its time in places proportional to the density of the posterior distribution. 
- that is, θ is a draw from the posterior distribution.
