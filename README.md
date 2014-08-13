rank-k-encoding
===============

_Goal_: fitting neural encoding models with input nonlinearities

_Models fit_: linear, bilinear, rank-2, full-rank

## Example

### Stimulus

Simulate a 1-d stimulus for 2000 time bins. Stimulus values are i.i.d., and normally distributed with mean 5, variance 1.

![Stimulus](/images/stim.png?raw=true "Stimulus")

### Filters

Simulate a neuron with two input nonlinearities, _f1_ and _f2_, and two stimulus history filters, _w1_ and _w2_. We can model the input nonlinearities by projecting them onto a series of Gaussian bumps spanning the range of stimulus values. The nonlinearities are now represented by two sets of basis weights, _b1_ and _b2_.

![Filter A](/images/resp-1.png?raw=true "1st filter")
![Filter B](/images/resp-2.png?raw=true "2nd filter")

The spike rate _r(t)_ is calculated as follows, where X(t) is a vector of the last 8 stimulus values, and e is gaussian noise with mean 250, variance 1:

r(t) = w1' b1 f1(X(t)) + w2' b2 f2(X(t)) + e

Or, letting C1 = w1' b1, and C2 = w2' b2 (shown in the images above), then:

r(t) = C1 X(t) + C2 X(t) + e

### Response

Here's the resulting simulated spike rate over time:

![Response](/images/rate.png?raw=true "Response")


And here's a plot of the spike rate (response) vs. the stimulus value at that time. The red line is the base spike rate plus gaussian noise (_e_ in the definition of r(t) above). Note the lack of a linear relationship between stimulus value and spike rate, a result of the rate being a function of the stimulus' history. Also, the spike rate shows the most variance when the stimulus is 0.

![Stimulus vs. Response](/images/stim-v-rate.png?raw=true "Stimulus vs. Response")

### Fitting methods

We want to fit the neuron's response, r(t), by finding the set of weights on the stimulus history and input nonlinearity that minimize our squared error. In other words, we want a rh(t) such that (rh(t) - r(t)).^2 is as small as possible.

#### Linear

This finds a set of spike history weights, wl, such that:

rh(t) = wl' X(t) + e

In other words, it assumes there is no input nonlinearity.

#### bilinear (aka rank-1)

Finds a set of spike history weights, wU, and a set of basis weights, bU, such that:

rh(t) = wU' bU X(t) + e

In other words, it assumes there is only one input nonlinearity.

#### rank-2

Finds two sets of spike history weights, wU and wV, and two sets of basis weights, bU and bV, such that:

rh(t) = wU' bU X(t) + wU' bU X(t) + e

This is identical to the method by which we simulated spike rates.

#### full-rank

Finds two matrices MU and MV such that:

rh(t) = MU X(t) + MV X(t) + e

Note how this model has _more_ parameters than our simulated model.

### Fitting results: Rates

The linear and bilinear models both do a pretty poor job.

![Linear model](/images/rate-linear.png?raw=true "Linear model")
![biinear model](/images/rate-bilinear.png?raw=true "Bilinear model")

The rank-2 and full-rank models do nearly perfectly, which is expected since the simulated neuron explicitly uses a rank-2 model. (A full-rank model can capture the effect of any rank-k model.)

![Rank-2 model](/images/rate-best.png?raw=true "Rank-2 model")

### Fitting results: Residuals

Error in spike rate prediction as a function of response value:

![Residuals](/images/residuals.png?raw=true "Residuals")

Here's the same plot without the linear and bilinear models:

![Best residuals](/images/residuals-best.png?raw=true "Best residuals")
