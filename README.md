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
import numpy as np

# Create simulation time
sim_time = tn.SimulationTime(PST=24, MST=48, dt=300)

# Create weather data
weather = tn.Weather(
    year=2023, month=7, day=1, local_hour=0, local_min=0, local_sec=0,
    local_latitude=37.5, local_longitude=127.0, standard_longitude=135.0,
    temp=tn.C2K(np.random.rand(sim_time.tN+1) * 10 + 20),
    Vz=np.random.rand(sim_time.tN+1) * 5,
    Gh=np.random.rand(sim_time.tN+1) * 1000
)

# Create indoor air
indoor_air = tn.IndoorAir(temperature=tn.C2K(20), volume=100, ACH=0.5)

# Create layers
concrete = tn.SetLayer(L=0.2, dx=0.02, k=1.4, c=880, rho=2300)
insulation = tn.SetLayer(L=0.1, dx=0.02, k=0.04, c=1000, rho=30)

# Create construction
wall = tn.SetConstruction(
    name="Wall", 
    layers=[concrete, insulation], 
    area=10, 
    roughness="medium rough", 
    azimuth=0, 
    tilt=90, 
    Tinit=tn.C2K(20)
)

# Run simulation
tn.structure_simulation([wall], sim_time, weather, indoor_air, "output_folder")

# Use functions
temp_celsius = 25
temp_kelvin = tn.C2K(temp_celsius)

# unit
tn.J2kJ
tn.d2h
