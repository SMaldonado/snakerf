# snakerf

A mediocre ~~collection of RF/analog circuits scripts~~ Python package someday when I get around to [releasing](https://packaging.python.org/tutorials/packaging-projects/) it as one. If you can still read this line in the README, please don't judge the actual software content of this repository too much.

The intent of this package is to:
- Facilitate the design of VHF/UHF digital radio systems
- Fill a perceived gap in [scikit-rf](http://scikit-rf.org), which abstracts RF circuits into matrices of S/X/Y/Z parameters and is poorly suited to tasks like designing matching networks (it _can_, and they have example code showing how, but that doesn't mean it's well suited to the task)
- Make up for (LT)Spice's poor handling of scattering parameters and general clunkiness
- Stop rewriting the same code repeatedly and then losing/forgetting about it/losing access to the iPython server it ran on
- Maintain my employability

## Things that currently "work":
- Flaky 2-port modeling, including "arbitrary" networks of passive components and _very_ flaky transmission lines
- Approximately modeling microstrips and striplines at sufficiently low frequencies
- Creating time domain representations of FSK, PSK, and MSK modulated signals
- Modeling additive white Gaussian noise and non-white background environmental noise
- Converting between time-domain voltages and voltage/power spectra
- Converting between decibels and linear quantities
- Generating [Gold codes](https://en.wikipedia.org/wiki/Gold_code) up to length 2^15 + 1 (after which my laptop started throwing memory errors; your mileage may vary)
- Demodulators for FSK and PSK

## Real goals
- Demodulator for MSK
- Mixer/amplifier modeling with noise
- Tool for designing filters using purchaseable component values
- Clean interface to simulate an entire signal path and model BER performance
- Actual thorough test code
- Sustained commit streak

## Aspirational goals
- Cython accelerated math
- More nuanced transmission line modeling i.e. 2.5D simulation
- Parametrically generating RF structures for PCBs
- Mixer/amplifier nonlinearity modeling
- Modulator/demodulator for QAM
- Simulate FHSS
- Playing nicely with scikit-rf
- Importing/exporting SPICE netlists
- This code actually being useful to someone else

## Known issues
- "Power" spectra are very easy to use incorrectly; considering some significant restructuring
- Despite now having lots of little functional blocks, interfaces between different pieces of code are currently bad
- There's some evidence that noise modeling isn't quite right, possibly for subtle numerical reasons
- Modulation should use Gray code
