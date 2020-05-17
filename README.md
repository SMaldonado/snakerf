# snakerf

A mediocre ~~collection of RF/analog circuits scripts~~ Python package someday when I get around to [releasing](https://packaging.python.org/tutorials/packaging-projects/) it as one. If you can still read this line in the README, please don't judge the actual software content of this repository too much.

The intent of this package is to:
- Facilitate the design of VHF/UHF digital radio systems
- Fill a perceived gap in scikit-rf, which abstracts RF circuits into matrices of S/X/Y/Z parameters and is poorly suited to tasks like designing matching networks
- Make up for (LT)Spice's poor handling of scattering parameters and general clunkiness
- Stop rewriting the same code repeatedly and then losing/forgetting about it/losing access to the iPython server it ran on

## Things that currently "work":
- Modeling the frequency response of networks of passive components (basically anything that can be represented as a ladder of impedances)
- Creating time domain representations of FSK, PSK, and MSK modulated signals, as well as additive white Gaussian noise
- Converting between time-domain voltages and voltage/power spectra
- Converting between decibels and linear quantities
- Generating [Gold codes](https://en.wikipedia.org/wiki/Gold_code) up to length 2^15 + 1 (after which my laptop started throwing memory errors; your mileage may vary)

## Real goals
- Demodulators for FSK, PSK, and MSK
- Mixer/amplifier modeling with noise
- Tool for designing filters using purchaseable component values
- Modeling simple PCB features (microstrips)
- Clean interface to simulate an entire signal path and model BER performance
- Actual thorough test code
- Sustained commit streak

## Aspirational Goals
- Completely ransacking Saturn PCB's board modeling features
- Parametrically generating RF structures for PCBs
- Mixer/amplifier nonlinearity modeling
- Modulator/demodulator for QAM
- Simulate FHSS
- This code actually being useful to someone else
