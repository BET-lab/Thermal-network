import numpy as np
import pandas as pd
import convection as cv
import radiation as rd
import constant as c
from tqdm import tqdm
import thermal_network as tn

# Example usage
# Create simulation time
sim_time = tn.SimulationTime(PST=0, MST=1, dt=100)

# Create weather data (example values, replace with real data)
weather = tn.Weather(year=2023, month=7, day=1, local_hour=0, local_min=0, local_sec=0,
                    local_latitude=37.5, local_longitude=127.0, standard_longitude=135.0,
                    temp = tn.C2K(np.random.rand(sim_time.tN+1) * 10 + 20),  # Random temperatures between 20-30°C
                    Vz = np.random.rand(sim_time.tN+1) * 5,  # Random wind speeds between 0-5 m/s
                    Gh = np.random.rand(sim_time.tN+1) * 100)  # Random solar radiation between 0-1000 [W/m2]

# Create indoor air
indoor_air = tn.IndoorAir(temperature=tn.C2K(20), volume=100, ACH = 0.1)

# Create layers
concrete = tn.SetLayer(L=0.2, dx=0.02, k=1.4, c=880, rho=2300)
insulation = tn.SetLayer(L=0.1, dx=0.02, k=0.04, c=1000, rho=30)

# Create constructions
wall = tn.SetConstruction(name="Wall", layers=[concrete, insulation], area=10, 
                        roughness="medium rough", azimuth=0, tilt=90, Tinit=tn.C2K(20))
roof = tn.SetConstruction(name="Roof", layers=[concrete, insulation], area=20, 
                        roughness="rough", azimuth=0, tilt=0, Tinit=tn.C2K(20))

# Run simulation
tn.run_building_exergy_model([wall, roof], sim_time, weather, indoor_air, "output_folder")