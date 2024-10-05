import numpy as np
import pandas as pd
import convection as cv
import radiation as rd
import constant as c
from tqdm import tqdm
import thermal_network as tn

# Run simulation (fully unsteady model)
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

tn.run_building_exergy_model_fully_unsteady([wall, roof], sim_time, weather, indoor_air, "output_folder")



# Run simulation (single capacitance model)
sim_time = tn.SimulationTime(PST=0, MST=24.0, dt=10)

# Define weather data
weather_data = tn.Weather(
    year=2023,
    month=1,
    day=1,
    local_hour=0,
    local_min=0,
    local_sec=0,
    local_latitude=37.7749,
    local_longitude=-122.4194,
    standard_longitude=-120,
    temp=np.random.uniform(low=tn.C2K(0), high=tn.C2K(20), size=sim_time.tN),
    Vz=np.random.uniform(low=0, high=5, size=sim_time.tN),
    Gh=np.random.uniform(low=0, high=800, size=sim_time.tN)
)

# Define indoor air conditions
indoor_air = tn.IndoorAir(
    temperature=tn.C2K(20),  # 20°C in Kelvin
    volume=100.0,  # 100 m3
    ACH=0.1  # Air changes per hour
)

# Define building structure
wall =  tn.SetSingleCapacitanceComponent(
        name="Wall",
        roughness="medium rough",
        L=0.2,  # 20 cm
        k=0.5,  # W/mK
        c=1000,  # J/kgK
        rho=2200,  # kg/m3
        area=10.0,  # m2
        azimuth=180,  # South-facing
        tilt=90,  # Vertical
        Tinit=tn.C2K(20)  # Initial temperature 20°C in Kelvin
    )
structure = [wall]

# Define file path for output
output_filepath = "../output"

# Run the building exergy model
tn.run_building_exergy_model_single_capacitance(
    Structure=structure,
    SimulationTime=sim_time,
    Weather=weather_data,
    IndoorAir=indoor_air,
    filepath=output_filepath
)

