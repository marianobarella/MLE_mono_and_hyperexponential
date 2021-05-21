# MLE_mono_and_hyperexponential

Finding the kinetic binding time of an imager probe in a DNA-PAINT experiment
---
Simulation of the data with exponential distributions using `np.random` module. Two cases considered: 

1. Monoexponential case: single exponential distribution, only one kind of binding event.
    
2. Hyperexponential case: two exponential distributions, two possible binding events with a long and short lifetime. The ratio between their probabilities is defined by the size of the samples. See Section 1.2.5 of this [document](https://www.win.tue.nl/~iadan/sdp/h1.pdf) for the definition of hyperexponential distribution.  
    
Simulated data is binned and plotted in a histogram with some criteria. Linear fitting of the histogram is performed in both cases. Maximum Likelihood Estimation (MLE) is performed to estimate the best parameters of Mono and Hyper distributions in each case, respectively. Results of the MLE are plotted with the simulated data to observe how well they match. Instead of maximizing the Likelihood, the -log() of the Likelihood is minimized using `scipy.optimize.minimize`.
