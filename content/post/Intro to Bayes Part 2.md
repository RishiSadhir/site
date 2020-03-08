---
title: "Intro to Bayes: Part 2"
author: "Rishi Sadhir"
date: '2020-03-08'
categories: ["Python", "Bayesian", "Statistics"]
tags: ["Python", "Bayesian", "Statistics"]
---

In this post, we'll continue to use the coin flip example from [part 1](https://www.rishisadhir.com/2020/02/08/intro-to-bayes-part-1/). Recall that we are interested in the posterior distribution of the parameter θ, which is the probability that a coin toss results in “heads”. Our prior distribution is an uninformative beta distribution with parameters 1 and 1. We also used a binomial likelihood function to estimate how representative a candidate value for θ is to have generated the data we have on hand - 3 observed heads out of 9 tosses.

In the previous post, we relied on the fact that we could calculate the posterior distribution analytically. We solved for the posterior distribution by taking advantage of conjugacy, when the prior and the posterior have the same distribution. This makes calculating the posterior really easy but we don't have the opportunity to use this trick often in practice. In this post we'll level up by starting to use MCMC simulation to sample from our target distribution: The posterior distribution of θ. In the real world, we'd then use these samples to make decisions under the assumption that the samples approximate the target distribution in high enough number.

# Markov Chain Monte Carlo
We'll use MCMC with the M–H algorithm to generate a sample from the posterior distribution of 𝜃 . There are three basic parts to this technique:

1. Proposal Distribution - Monte Carlo
2. Proposal Exploration - Markov Chain
3. Proposal Selection - MH algorithm


```python
# Numeric computing
import numpy as np
import scipy.stats as stats
import pymc3 as pm
# Plotting
import matplotlib.pyplot as plt
```


```python
# Plot theme
COLOR = "#2A0933"
plt.rcParams['text.color'] = COLOR
plt.rcParams['text.hinting_factor'] = 8
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['axes.facecolor'] = "eeeeee"
plt.rcParams['axes.edgecolor'] = "bcbcbc"
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.titlesize'] = "x-large"
plt.rcParams['axes.labelsize'] = "large"
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.rcParams['grid.color'] = COLOR  
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = .7
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['patch.linewidth'] = 0.5
plt.rcParams['patch.facecolor'] = "blue"
plt.rcParams['patch.edgecolor'] = "eeeeee"
plt.rcParams['patch.antialiased'] = True  
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['legend.fancybox'] = True
plt.rcParams['figure.figsize'] = (11, 8)
plt.rcParams['figure.dpi'] = 300
```

## 1. Proposal Distribution - Monte Carlo

MCMC needs a way to explore the parameter space, which is to say we need to be able to test out different possible values for θ. One way to do this is to perform Monte Carlo simulation. 

The term “Monte Carlo” refers to methods that rely on the generation of random numbers. In the example below, we will draw random numbers from a proposal distribution. In practice, the choice of distribution is given much thought, but here we'll use Normal(.5, .1). There are a couple good justifictions for this. 

1. I'd expect this to be pretty close to the target distribution we're after. 
2. A normal distribution is symmetric on both sides of the mean which make some of the math we'll see later on easier. 

Below, we generate a series of random numbers from our proposal distribution. The graph on the left is called a *trace plot*. Trace plots display the values of θ in the order in which they are drawn. It basically shows our parameter exploratin pattern. The plot on the left combines all these samples in to a histogram to tell us where to allocate our belief of θs location.


```python
N = 10_000
x = stats.norm.rvs(loc=.5, scale=.1, size=N)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,3), gridspec_kw={'width_ratios': [3, 1.5]})
axes[0].set_ylim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("")
axes[1].set_title("Density")
axes[0].set_xlabel("MCMC Iteration")
axes[0].set_title("Monte Carlo Exploration")
axes[0].yaxis.tick_right()
axes[1].set_ylabel("$\\theta$")
axes[1].set_title("Density")
axes[0].set_xlabel("Sample number")
axes[0].set_title("Monte Carlo trace plot")
axes[1].hist(x, bins = 30, color = COLOR, orientation="horizontal")
xax = np.linspace(start=0,stop=len(x)-1,num=len(x))
axes[0].plot(xax,x,color=COLOR, linewidth=1)
plt.show()
```

![png](/post/intro_bayes_2/figure-html/output_4_0.png)


The above plot tells us a few import things. 

1. The proposal distribution is stationary. It doesn't change at all as we continue to draw samples. 
2. The more samples we draw, the better shape our density takes. With more samples on the left, the more the density on the right approximates our proposal distribution.

This is cool and all, but we want a distribution that approximates the posterior, not the proposal distribution. We need to go another step forward to explore the parameters space better.

## 2. Proposal Exploration - Markov Chain
A Markov chain is a sequence of numbers where each number is dependent on the previous number in the sequence. For example, we could draw values of θ from a normal proposal distribution with a mean equal to the previous value of θ. 

Thats exactly what we do below. We start with a random value from the proposal distribution: 
\begin{align}
\theta_i \sim normal(.5, .1)
\end{align}

Every subsequent draw uses the previous draw as its mean 

\begin{align}
\theta_i \sim normal(\theta_{i-1}, .1)
\end{align}

We draw 10,000 random values of θ using a markov chain to see how this process works.


```python
N = 1000
x = [None] * N

# Initialize our chain with a random value
x[0] = stats.norm.rvs(loc=.5, scale=.1, size=1)[0]

# Add each movement away from the previous 
# value to the chain 
for i in range(1,N):
    prev = x[i-1]
    x[i] = stats.norm.rvs(loc=prev, scale=.1, size=1)[0]

# Plot the results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,3), gridspec_kw={'width_ratios': [3, 1.5]})
axes[1].set_xlabel("")
axes[1].set_title("Density")
axes[0].set_xlabel("MCMC Iteration")
axes[0].set_title("Markov Chain Exploration")
axes[0].yaxis.tick_right()
axes[1].set_ylabel("$\\theta$")
axes[1].set_title("Density")
axes[0].set_xlabel("MCMC Iteration")
axes[1].hist(x, bins = 30, color = COLOR, orientation="horizontal")
xax = np.linspace(start=0,stop=len(x)-1,num=len(x))
axes[0].plot(xax,x,color=COLOR, linewidth=1)
fig.tight_layout()
plt.show()
```


![png](/post/intro_bayes_2/figure-html/output_6_0.png)


The plots above show us two differences between Monte Carlo and Markov Chains. 

1. The proposal distribution is changing with each iteration. This creates a trace plot with a “random walk” pattern - the variability is not the same over all iterations. 
2. The resulting density plot does not look like the proposal distribution or any other useful distribution. It certainly doesn’t look like a posterior distribution.

Again, this isn't what we want, but we are getting a little bit closer. We've figured out a way to eplore the parameter space, but now we need to think about how to make this exploration look like the posterior. We could improve our sample by keeping proposed values of θ that are more likely under the posterior distribution and discarding values that are less likely. How should we do this? One answer is the M–H algorithm!

## Proposal Selection - MH algorithm

MH is an algo that allows us to sample from a generic probability distribution even if we don't know the normalizing constant. To do this, we construct and sample from a markov chain whose stationary distribution is the target distribution we are looking for. It consists of picking an arbitrary starting value and then iteratively accepting or rejecting candidate samples drawn from another distribution, one that is easy to sample. 

Lets say we want to produce samples from a target distribution p(θ) but we only know it up to proportionality. This is the exact case we have for the posterior distribution.

$$
Pr(\theta | data) = \frac{Pr(data) | \theta) \times Pr(\theta)}{Pr(data)} \propto Pr(data | \theta) \times Pr(\theta)
$$

The algorithm procedes as follows:
![png](/post/intro_bayes_2/figure-html/algo.png)
        
Steps 2.C.b and 2.C.c act as a correction since the proposal distribution is not the target distribution. At each step in the chain, we draw a candidate and decide whether to move the chain there or stay where we are. If the move is advantageous, we will move there for sure. If it isn't advantageous we might still move there, but only with probability α. 

I spell these steps out in the code below too. Lets give it a spin!



```python
def proposal_density(theta):
    """
    Calculate a value proportional to the posterior
    for a proposed value of the parameter
    """
    prior = stats.beta.pdf(theta, 1, 1)
    likelihood = stats.binom.pmf(p=theta, k=3, n=9)
    return prior * likelihood
    
N = 20_000
x = [None] * N
# Step 1 - Initialize chain
x[0] = stats.norm.rvs(loc=.5, scale=.1, size=1)[0]
# Step 2 - Draw samples
for i in range(1,N):
    # Step 2.A - Propose new value
    previous = x[i-1]
    proposal = stats.norm.rvs(loc=previous, scale=.1, size=1)[0]
    # Step 2.B - Calculate improvement
    improvement_ratio = proposal_density(proposal)/proposal_density(previous)
    # Step 3.C - Accept or reject proposal
    if improvement_ratio >= 1:
        x[i] = proposal
    elif 0 < improvement_ratio < 1:
        if stats.bernoulli.rvs(improvement_ratio, size = 1)[0]:
            x[i] = proposal
        else:
            x[i] = x[i-1]
    else:
        x[i] = x[i-1]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,3), gridspec_kw={'width_ratios': [3, 1.5]})
axes[0].set_ylim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("")
axes[1].set_title("Density")
axes[0].set_xlabel("MCMC Iteration")
axes[0].set_title("Monte Carlo Exploration")
axes[0].yaxis.tick_right()
axes[1].set_ylabel("$\\theta$")
axes[1].set_title("Density")
axes[0].set_xlabel("MCMC Iteration")
axes[0].set_title("HMC NUTS trace plot")
axes[1].hist(x, bins = 30, color = COLOR, orientation="horizontal")
xax = np.linspace(start=0,stop=len(x)-1,num=len(x))
axes[0].plot(xax,x,color=COLOR, linewidth=1)
plt.show()
```


![png](/post/intro_bayes_2/figure-html/output_9_0.png)


A few things to note again. First, the proposal distribution changes with most iterations. Second, the trace plot does not exhibit the random walk pattern we observed using MCMC alone. The variation is similar across all iterations, And finally, the density plot looks like a useful distribution.

Lets check our posterior samples against what we calculated analytically in the previous post.




```python
x_seq = np.linspace(0, 1, 1000)
_density = stats.gaussian_kde(x)
y_post = _density.pdf(x_seq)
y_analytic = stats.beta.pdf(x_seq, 3+1, 9-3+1)

fig, ax = plt.subplots(figsize=(14,3))
ax.plot(x_seq, y_post, color = "#2D718E", label = "MH Posterior")
ax.plot(x_seq, y_analytic, color = "#73D055", label = "Conjugate Posterior")
ax.set_ylabel("Density")
ax.set_xlabel("$\\theta$")
plt.legend()
plt.suptitle("Conjugate and MH Posterior samples align well")
plt.show()
```


![png](/post/intro_bayes_2/figure-html/output_11_0.png)


# Don't use Metropolis Hastings
MH is the simplest and least reliable way of doing MCMC. The algorithms is extremely simple! It approximates a target distribution by accepting or rejecting random proposals. If you make proposals correctly and accept proposals under specific conditions, the accepted parameter values will comprise samples from the target distribution.

There have been many improvements since M-H was first proposed in 1953. MH wastes a lot of time exploring inefficiently due to its random nature. If there’s a random way to do something, there’s usually a less random way that is both better and requires more thought. Bayesian tools today use a really cool algorithm called HMC NUTs that essentially treats the parameters space like a physics algorithm. Your chains skate around the paramters space with 0 friction. Usually, this parameter space is bowl shaped (in log space) so all you have is gravity pulling you toward high probablility density regions.

There are a lot of excellent resources out there to learn the principles behind NUTS, but a great place to start is [this blog post by Richard McElreath](https://elevanth.org/blog/2017/11/28/build-a-better-markov-chain/). If you grasp the basics of M-H, its really not necessary to go deep on the newer stuff to use it in practice.

### PyMC3
This post goes in to detail to be instructive... Don't go implementing your own parameter sampling algorithms. There are so many great ones out there already! Stan is my favorite but PyMC3 is a great one too. With just a couple lines of code, I can unleash the physics simulation on our coin flipping example. So lets do that to show how easy it is.

The first step, like before, is to collect some data.


```python
obs_coin_flips = np.array([1]*3 + [0]*6)
obs_coin_flips
```




    array([1, 1, 1, 0, 0, 0, 0, 0, 0])



Now we just define our model, exactly like we did in the first blog post, but this time in code. 


```python
with pm.Model() as coin_flip:
    pr = pm.Beta("pr", 1, 1)
    is_head = pm.Bernoulli("is_head", pr, observed = obs_coin_flips)
    
coin_flip
```




$$
            \begin{array}{rcl}
            \text{pr} &\sim & \text{Beta}(\mathit{alpha}=1.0,~\mathit{beta}=1.0)\\\ ishead &\sim & \text{Bernoulli}(\mathit{p}=\text{pr})
            \end{array}
            $$


With our model definition in hand, we tell PyMC3 to start sampling -- And off goes four markov chains in parallel. PyMC3 figures out initial values and a good proposal distribution on its own. It then spins up 4 markov chains that explore according to HMC NUTS.



```python
with coin_flip:
    # Sample from the posterior distribution
    posterior = pm.sample(draws = 2000, chains = 4, cores = 4)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [pr]
    Sampling 4 chains, 0 divergences: 100%|██████████| 10000/10000 [00:01<00:00, 7253.96draws/s]


And just like that, we have samples from the posterior distribution. Finally, we plot them to make sure they look like they did in our examples above.


```python
x = posterior["pr"]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,3), gridspec_kw={'width_ratios': [3, 1.5]})
axes[0].set_ylim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("")
axes[1].set_title("Density")
axes[0].set_xlabel("MCMC Iteration")
axes[0].set_title("Monte Carlo Exploration")
axes[0].yaxis.tick_right()
axes[1].set_ylabel("$\\theta$")
axes[1].set_title("Density")
axes[0].set_xlabel("MCMC Iteration")
axes[0].set_title("HMC NUTS trace plot")
axes[1].hist(x, bins = 30, color = COLOR, orientation="horizontal")
xax = np.linspace(start=0,stop=len(x)-1,num=len(x))
axes[0].plot(xax,x,color=COLOR, linewidth=1)
plt.show()
```


![png](/post/intro_bayes_2/figure-html/output_19_0.png)


# Summary

This blog post introduced the idea behind MCMC using the M–H algorithm. Note that I have omitted some details and ignored some assumptions so that we could keep things simple and develop our intuition. Stan and PyMC3 implement much more sophisticated algorithms and should be your go-to... But the basic idea is the same. Go forth without fear and let your chains run wild! I hope I’ve inspired you to leverage Bayesian methods in your next analysis.
