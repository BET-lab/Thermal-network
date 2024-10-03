# Thermal-network
This files attribute the function and class for 1 dimensional thermal network analysis

## Installation
```
pip install git+https://github.com/BET-lab/Thermal-network.git
```

## Usage

Here's a basic example of how to use the library:

```python
import thermal_network as tn

# Create layers
layer1 = tn.SetLayer(L=0.1, dx=0.01, k=0.5, c=1000, rho=2000)
layer2 = tn.SetLayer(L=0.2, dx=0.02, k=1.0, c=800, rho=2500)

# Create construction
wall = tn.SetConstruction(layer1, layer2)

# Use functions
temp_celsius = 25
temp_kelvin = tn.D2K(temp_celsius)

# unit
tn.J2kJ
tn.d2h
```
