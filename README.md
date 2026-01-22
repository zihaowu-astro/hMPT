Purpose
-------

This package optimizes JWST/NIRSpec MSA configurations (RA, Dec, PA) to maximize the number of sources placed within shutters. 

This package is based on [eMPT](https://github.com/esdc-esac-esa-int/eMPT_v1) reimplemented in Python. We have several extensions:

1. Flexiable in-shutter criteria. We have two options: a) source centering options as in MPT; b) an in-shutter flux threshold that accounts for the PSF and source fluxes.
2. Optimize top solutions of grid search using a differential evolution algorithm.
3. A flexible criterion for handling shutter conflicts.


Authors
-------
Daniel Eisenstein, Samuel McCarty, Zihao Wu
