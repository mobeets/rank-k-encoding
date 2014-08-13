rank-k-encoding
===============

fitting neural encoding models with input nonlinearities

models: linear, bilinear, rank-2, full-rank

## Example

### Stimulus

Stimulus is a 1-d value at each time-point, normally distributed with mean 5, variance 1.

![Stimulus](/images/stim.png?raw=true "Stimulus")

### Filters

Simulate a neuron's spike rate as a rank-2 response to the following filters, plus gaussian noise:

![Filter A](/images/resp-1.png?raw=true "1st filter")
![Filter B](/images/resp-2.png?raw=true "2nd filter")

The neuron's spike rate in this case is a function of the last 8 stimulus values.

### Response

Simulated spike rate over time:

![Response](/images/rate.png?raw=true "Response")

Spike rate vs. stimulus value:

![Stimulus vs. Response](/images/stim-v-rate.png?raw=true "Stimulus vs. Response")

### Fitting results: Rates

The linear and bilinear models do a pretty poor job. The rank-2 and full-rank do nearly perfectly, which is expected since the simulated neuron explicitly uses a rank-2 model.

![Linear model](/images/rate-linear.png?raw=true "Linear model")
![biinear model](/images/rate-bilinear.png?raw=true "Bilinear model")
![Rank-2 model](/images/rate-best.png?raw=true "Rank-2 model")

### Fitting results: Residuals

Error in spike rate prediction as a function of response value:

![Residuals](/images/residuals.png?raw=true "Residuals")

Here's the same plot without the linear and bilinear models:

![Best residuals](/images/residuals-best.png?raw=true "Best residuals")
