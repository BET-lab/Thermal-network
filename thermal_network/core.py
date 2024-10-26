import constant as c
import radiation as rd
import convection as cv

# from .import constant as c
# from .import radiation as rd
# from .import convection as cv

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from dataclasses import dataclass

### 1. Helper Functions

## 1.1 Temperature Conversion Functions
def C2K(temp_C: float) -> float:
    return temp_C + 273.15

def K2C(temp_K: float) -> float:
    return temp_K - 273.15

def F2C(temp_F: float) -> float:
    return (temp_F - 32) * 5 / 9

def C2F(temp_C: float) -> float:
    return temp_C * 9 / 5 + 32

## 1.2 Unit Conversion Functions
def cm2in(cm: float) -> float:
    return cm / 2.54

## 1.3 Array and DataFrame Conversion Functions
def arr2df(arr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(arr)

def df2arr(df: pd.DataFrame) -> np.ndarray:
    return df.values

## 1.4 Time Interval Calculation Functions
def half_time_vector(vec: np.ndarray) -> np.ndarray:
    return (vec[:-1] + vec[1:]) / 2

def half_time_matrix(mat: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        return (mat[:-1, :] + mat[1:, :]) / 2
    elif axis == 1:
        return (mat[:, :-1] + mat[:, 1:]) / 2
    else:
        raise ValueError("Axis must be 0 or 1")

## 1.5 Data Interpolation Function
def interpolate_hourly_data(hourly_data: np.ndarray, time_step: float) -> np.ndarray:
    rN = len(hourly_data)
    x = np.arange(rN)
    interval = time_step * c.s2h
    x_new = np.arange(0, rN + interval, interval)
    return np.interp(x_new, x, hourly_data)


### 2. Data Classes

## 2.1 Simulation Time Class
@dataclass
class SimulationTimeParameters:
    PST: float  # Pre-simulation time [hr]
    MST: float  # Main-simulation time [hr]
    dt: float   # Time step [s]
    start_time: pd.Timestamp

    def __post_init__(self):
        # Calculate simulation times
        self.TST = self.PST + self.MST  # Total-simulation time [hr]
        self.end_time = self.start_time + pd.Timedelta(hours=self.TST)
        self.time_step = pd.Timedelta(seconds=self.dt)
        
        # Generate time range
        self.time_range = pd.date_range(start=self.start_time, end=self.end_time, freq=self.time_step)
        
        # Calculate time steps
        self.tN = len(self.time_range)
        self.ts_PST = int(self.PST * c.h2s / self.dt)  # Number of pre-simulation time steps
        self.ts_MST = int(self.MST * c.h2s / self.dt)  # Number of main-simulation time steps
        self.ts_TST = self.ts_PST + self.ts_MST  # Number of total-simulation time steps
        
        # Generate time arrays
        self.ts_s = np.arange(0, self.tN) * self.dt  # time step array [s]
        self.ts_m = self.ts_s * c.s2m  # time step array [min]
        self.ts_h = self.ts_s * c.s2h  # time step array [hr]

    def get_pre_simulation_range(self):
        return self.time_range[:self.ts_PST]

    def get_main_simulation_range(self):
        return self.time_range[self.ts_PST:]

    def get_time_index(self, time):
        return self.time_range.get_loc(time)

 
## 2.2 Weather Data Class
@dataclass
class WeatherData:
    location: str
    temp: np.ndarray
    Vz: np.ndarray
    DNI_H: np.ndarray
    DHI: np.ndarray
    standard_longitude: float = 135 # standard longitude of Korea

    def __post_init__(self):
        self.GHI = self.DNI_H + self.DHI


## 2.3 Indoor Air Class
@dataclass
class IndoorAir:
    ACH: float # Air change rate [1/hr]
    h_ci: float # convective heat transfer coefficient [W/m2K]
    volume: float # Volume of indoor air [m3]
    specific_heat_capacity: float = 1005 # specific heat capacity of air [J/kgK]
    density: float = 1.225 # density of air [kg/m3]
    temperature: float = C2K(20) # initial temperature [K]

    def __post_init__(self):
        self.C = self.specific_heat_capacity * self.density  # Volumetric heat capacity [J/m3K]

    def temp_update(self, heat_gain: float, Toa: float, dt: float):
        # Calculate indoor air temperature update
        dT_by_ACH = self.C * (self.volume * self.ACH / c.h2s) * (Toa - self.temperature) * dt /(self.volume * self.C) # [K]
        dT_by_heat_gain = heat_gain / (self.volume * self.C) # [K]
        self.temperature += dT_by_ACH + dT_by_heat_gain

## 2.4 Building Material Layer Class
@dataclass
class Layer:
    L: float  # Length [m]
    dx: float # dx [m]
    k: float  # Thermal conductivity [W/mK]
    c: float  # Specific heat capacity [J/kgK]
    rho: float # Density [kg/m3]

    def __post_init__(self):

        # Number of discretization
        self.discr_num = round(self.L / self.dx)  # Number of discretization [-]
        
        # Thermal properties of the layer
        self.C = self.c * self.rho  # Volumetric heat capacity [J/m3K]
        self.R = self.dx / self.k  # Thermal resistance [m2K/W]
        self.K = self.k / self.dx  # Thermal conductance # [W/m2K]
        self.alpha = self.k / self.C  # Thermal diffusivity [m2/s]

## 2.5 Building Structure Class
@dataclass
class Construction:
    '''This class is used to define the building structure
    추가할 수정사항
    1. Tinit을 설정하면 자동으로 첫 타임스텝의 온도를 설정하도록 수정'''
    name: str
    layers: List[Layer]
    roughness: str
    Tinit: float
    area: float
    azimuth: float
    tilt: float

    # Methods to calculate thermal properties of the structure
    def __post_init__(self):
        
        # Roughness check
        roughness_list = ['very rough', 'rough', 'medium rough', 'medium smooth', 'smooth', 'very smooth']
        if self.roughness not in roughness_list:
            raise ValueError(f"roughness must be one of {roughness_list}")
        
        # Calculate physical properties of the structure
        self.layer_discr_counts = [layer.discr_num for layer in self.layers]
        self.layer_num = len(self.layers)
        self.total_discr_num = sum(self.layer_discr_counts)
        self.thick = sum(layer.L for layer in self.layers)

        # Volumetric heat capacity [J/m3K]
        self.C = np.repeat([layer.C for layer in self.layers], self.layer_discr_counts)

        # Cell size and distance between node [m]
        self.dx = np.repeat([layer.dx for layer in self.layers], self.layer_discr_counts)
        self.dx_L = np.array([self.dx[0]/2] + [(self.dx[i-1] + self.dx[i])/2 for i in range(1, self.total_discr_num)])
        self.dx_R = np.array([(self.dx[i] + self.dx[i+1])/2 for i in range(self.total_discr_num-1)] + [self.dx[-1]/2])
        
        # Thermal resistance [m2K/W]
        self.R = np.repeat([layer.R for layer in self.layers], self.layer_discr_counts)
        self.R_L = np.array([self.R[0]/2] + [(self.R[i-1] + self.R[i])/2 for i in range(1, self.total_discr_num)])
        self.R_R = np.array([(self.R[i] + self.R[i+1])/2 for i in range(self.total_discr_num-1)] + [self.R[-1]/2])
        
        # Themal conductance [W/m2K]
        self.K = 1 / self.R
        self.K_L = 1 / self.R_L
        self.K_R = 1 / self.R_R

        # Total thermal properties
        self.R_tot = np.sum(self.R)
        self.K_tot = 1 / self.R_tot

    def get_thermal_capacity_per_area(self):
        return np.sum(self.C * self.dx) # [J/m2K]


## 2.6 Single Capacitance Component Class
@dataclass
class SingleCapacitanceComponent:
    ''' This component correspond to Construction Class'''
    name: str
    roughness: str
    L: float  # Thickness [m]
    k: float  # Thermal conductivity [W/mK]
    c: float  # Specific heat capacity [J/kgK]
    rho: float  # Density [kg/m3]
    area: float  # Area of component [m2]
    azimuth: float # Azimuth of component [deg]
    tilt: float # tilting angle of component [deg]
    Tinit: float # Initial temperature [K]

    def __post_init__(self):

        # Thermal properties of the layer
        self.C = self.c * self.rho  # Volumetric heat capacity [J/m3K]
        self.R = self.dx / self.k  # Thermal resistance [m2K/W]
        self.K = self.k / self.dx  # Thermal conductance # [W/m2K]
        self.V = self.area * self.L # Volume of the component [m3]


### 3. Simulation Functions

## 3.1 Data Processing Function
def eliminate_pre_simulation_data(data: np.ndarray, pre_simulation_time_step: int) -> np.ndarray:
    if data.ndim == 1:
        return data[pre_simulation_time_step:]
    elif data.ndim == 2:
        return data[pre_simulation_time_step:, :]

## 3.2 TDMA Calculation Function
def TDMA(Construction: Construction, T: np.ndarray, T_L: float, T_R: float, dt: float) -> np.ndarray:
    # TDMA (Tri-Diagonal Matrix Algorithm) calculation
    N = Construction.total_discr_num
    dx = Construction.dx
    K_L, K_R = Construction.K_L, Construction.K_R
    C = Construction.C
    
    a_list = -dt * K_L
    b_list = 2 * dx * C + dt * (K_L + K_R)
    c_list = -dt * K_R

    A = np.zeros((N, N))
    np.fill_diagonal(A[1:], a_list[1:])
    np.fill_diagonal(A, b_list)
    np.fill_diagonal(A[:, 1:], c_list[:-1])
    A_inv = np.linalg.inv(A)

    B = np.zeros((N, 1))
    B[0] = 2 * dt * K_L[0] * T_L + (2 * dx[0] * C[0] - dt * K_L[0] - dt * K_R[0]) * T[0] + dt * K_R[0] * T[1]
    B[1:-1,0] = dt * K_L[1:-1] * T[:-2] + (2 * dx[1:-1] * C[1:-1] - dt * K_L[1:-1] - dt * K_R[1:-1]) * T[1:-1] + dt * K_R[1:-1] * T[2:]
    B[-1,0] = dt * K_L[-1] * T[-2] + (2 * dx[-1] * C[-1] - dt * K_L[-1] - dt * K_R[-1]) * T[-1] + 2 * dt * K_R[-1] * T_R

    return np.dot(A_inv, B).flatten()

## 3.3 Solar Radiation Calculation Function
def calculate_DNI(weather: WeatherData, simulation_time_parameters: SimulationTimeParameters, construction_azi_arr: np.ndarray, construction_tilt_arr: np.ndarray) -> np.ndarray:
    '''각 construction의 연직 일사율을 2차원 배열(constructions, time)로 반환'''
    tN = simulation_time_parameters.tN
    dt = simulation_time_parameters.dt

    # Calculate solar radiation
    Normal_rad = np.zeros((len(construction_azi_arr), tN)) # [W/m2]
    
    for n in range(tN):
        for i, (azi, tilt) in enumerate(zip(construction_azi_arr, construction_tilt_arr)):
            year = simulation_time_parameters.time_range[n].year
            month = simulation_time_parameters.time_range[n].month
            day = simulation_time_parameters.time_range[n].day
            hour = simulation_time_parameters.time_range[n].hour
            minute = simulation_time_parameters.time_range[n].minute
            second = simulation_time_parameters.time_range[n].second
            
            Normal_rad[i][n] = rd.solar_to_unit_surface(DNI_H = weather.DNI_H[n], 
                                                        DHI = weather.DHI[n], 
                                                        station = weather.location, 
                                                        year=year, 
                                                        month=month, 
                                                        day=day, 
                                                        local_hour=hour,
                                                        local_min=minute, 
                                                        local_sec=second, 
                                                        surface_tilt=tilt, 
                                                        surface_azimuth=azi)
    return Normal_rad
    

## 3.4 Building exergy model simulation functions
def simulate_one_node_building_exergy(structure: List[Construction], simulation_time_parameters: SimulationTimeParameters,
                                                 weather: WeatherData, indoor_air: IndoorAir, filepath: str):
    # Simulation variables
    cN = len(structure) # compoenent number [-]
    dt = simulation_time_parameters.dt # time step [s]
    tN = simulation_time_parameters.tN # time step number [-]

    envelope_azi_arr = np.array([construction.azimuth for construction in structure])
    envelope_tilt_arr = np.array([construction.tilt for construction in structure])

    # Set outdoor and indoor conditions
    Toa = weather.temp.reshape(-1, 1) # outdoor air temperature [K] (reshape for broadcasting)
    Tia = np.full((tN,1), IndoorAir.temperature) # indoor air temperature [K]
    GHI = weather.GHI.reshape(-1, 1) # Global Horizontal Irradiance [W/m2] (reshape for broadcasting)

    # Calculate solar radiation
    Normal_rad = calculate_DNI(weather, simulation_time_parameters, envelope_azi_arr, envelope_tilt_arr)

    # Initialize matrices
    node_num = 3 # 0: Left node, 1: Center node, 2: Right node
    T = np.zeros((cN, tN, node_num)) # [component, time, node]
    q = np.zeros_like(T)
    Carnot_eff = np.zeros_like(T)

     # Set initial conditions
    for cidx, construction in enumerate(structure):
        T[cidx][0,:] = construction.Tinit
        q[cidx][0,:] = 0
        Carnot_eff[cidx][0,:] = 1 - Toa[0] / construction.Tinit

    # Convective heat transfer coefficients
    h_ci = indoor_air.h_ci
    R_ci = 1 / h_ci

    # Simulation time loop
    for n in tqdm(range(tN-1), desc="Simulation progress"):

        # Calculate heat gain and update indoor air temperature
        heat_gain = sum(h_ci * construction.area * (T[cidx][n,-1] - Tia[n,0]) for cidx ,construction in enumerate(structure)) * dt # heat gain by the component [W]
        indoor_air.temp_update(heat_gain, Toa[n+1,0], simulation_time_parameters.dt)
        Tia[n+1,0] = indoor_air.temperature

        # loop for each component
        for cidx, construction in enumerate(structure):

            # Thermal network variables
            h_co = cv.simple_combined_convection(construction.roughness, weather.Vz) # outdoor convection coefficient # [W/m2K]
            R_co = 1 / h_co # outdoor convection resistance # [m2K/W]
            R_half= construction.R_tot/2 # half resistance of component # [m2K/W]

            # Calculate center node temperature & heat flux
            T[cidx][n+1, 1] =  T[cidx][n,1] + (q[cidx][n,0] - q[cidx][n,-1]) * dt / (construction.get_thermal_capacity_per_area())
           
            # Calculate interface temperatures & heat fluxes (explicit method)
            T[cidx][n+1, 0] = (T[cidx][n+1,1] / R_half + Toa[n+1,0] / R_co[n+1] + Normal_rad[cidx][n+1]) / (1 / R_half + 1 / R_co[n+1]) # left surface
            T[cidx][n+1,-1] = (T[cidx][n+1,1] / R_half + Tia[n+1,0] / R_ci) / (1 / R_ci + 1 / R_half) # right surface
            
            q[cidx][n+1, 0] = (T[cidx][n+1,0] - T[cidx][n+1,1]) / (R_half)
            q[cidx][n+1,-1] = (T[cidx][n+1,1] - T[cidx][n+1,-1]) / (R_half)
            
            # Calculate heat fluxes
            q[cidx][n+1, 1] = (q[cidx][n+1,0] + q[cidx][n+1,-1]) / 2

            # Calculate Carnot efficiency
            Carnot_eff[cidx][n+1, :] = 1 - Toa[n+1,0] / T[cidx][n+1, :]

    # Post-processing
    CXcR = (construction.R_tot) * (Toa[:,0] * (q[:,:,1] / T[:,:,1])**2) # Exergy consumption rate [W/m3]

    # Component loop
    for i, construction in enumerate(structure):

        # Eliminate pre-simulation data (EPSD)
        T_EPSD = eliminate_pre_simulation_data(T[i], simulation_time_parameters.ts_PST)
        q_EPSD = eliminate_pre_simulation_data(q[i], simulation_time_parameters.ts_PST)
        Carnot_eff_EPSD = eliminate_pre_simulation_data(Carnot_eff[i], simulation_time_parameters.ts_PST)
        CXcR_EPSD = eliminate_pre_simulation_data(CXcR[i], simulation_time_parameters.ts_PST)
        Normal_rad_EPSD = eliminate_pre_simulation_data(Normal_rad[i], simulation_time_parameters.ts_PST)

        # Data framing
        columns = ["OS", "Middle", "IS"]
        time_index = simulation_time_parameters.time_range[simulation_time_parameters.ts_PST:].astype(str)

        T_df = pd.DataFrame(K2C(T_EPSD), columns=columns, index=time_index)
        q_df = pd.DataFrame(q_EPSD, columns=columns, index=time_index)
        CXcR_df = pd.DataFrame(CXcR_EPSD, columns=None, index=time_index)
        Carnot_eff_df = pd.DataFrame(Carnot_eff_EPSD, columns=columns, index=time_index)
        Normal_rad_df = pd.DataFrame(Normal_rad_EPSD, columns=["Solar Radiation [W/m2]"], index=time_index)

        # Export to Excel
        filename = f"{construction.name}.xlsx"
        with pd.ExcelWriter(f"{filepath}/{filename}") as writer:
            T_df.to_excel(writer, sheet_name='T')
            q_df.to_excel(writer, sheet_name='q')
            CXcR_df.to_excel(writer, sheet_name='XcR')
            Carnot_eff_df.to_excel(writer, sheet_name='Carnot_eff')
            Normal_rad_df.to_excel(writer, sheet_name='Normal_rad')
    
    # Post-processing 
    Tia_EPSD = eliminate_pre_simulation_data(Tia, simulation_time_parameters.ts_PST)
    Toa_EPSD = eliminate_pre_simulation_data(Toa, simulation_time_parameters.ts_PST)
    GHI_EPSD = eliminate_pre_simulation_data(GHI, simulation_time_parameters.ts_PST)
    environment_data = np.concatenate((K2C(Tia_EPSD), K2C(Toa_EPSD), GHI_EPSD), axis=1)

    # Data framing
    columns = ["Indoor Air Temperature [°C]", "Outdoor Air Temperature [°C]", "Global Horizontal Irradiance [W/m2]"]
    environment_df = pd.DataFrame(environment_data, columns=columns, index=time_index)

    # Export to Excel
    environment_df.to_excel(f"{filepath}/environment.xlsx")
    

def run_building_exergy_model_fully_unsteady(structure: List[Construction], simulation_time_parameters: SimulationTimeParameters, 
                         weather: WeatherData, indoor_air: IndoorAir, filepath: str):
    ''' 미완성 '''
    # Main simulation function
    
    # Initialize simulation
    num_constructions = len(structure)
    tN = simulation_time_parameters.tN
    dt = simulation_time_parameters.dt
    time_index = simulation_time_parameters.ts_h

    # Extract building structure parameters
    envelope_azi  = np.array([construction.azimuth for construction in structure])
    envelope_tilt = np.array([construction.tilt for construction in structure])

    # Set outdoor and indoor conditions
    Toa = weather.temp.reshape(-1, 1)
    Vz = weather.Vz
    Normal_rad = calculate_DNI(weather, envelope_azi, envelope_tilt, tN, dt)
    Tia = np.full((tN+1,1), indoor_air.temperature)

    # Initialize matrices
    T = np.zeros((num_constructions, tN + 1, max(construction.N for construction in structure)))
    T_L = np.zeros_like(T)
    T_R = np.zeros_like(T)
    q_in = np.zeros_like(T)
    q = np.zeros_like(T)
    q_out = np.zeros_like(T)
    Carnot_eff_L = np.zeros_like(T)
    Carnot_eff_R = np.zeros_like(T)

    # Set initial conditions
    for cidx, construction in enumerate(structure):
        T[cidx][0,:] = construction.Tinit
        T_L[cidx][0,:] = construction.Tinit
        T_R[cidx][0,:] = construction.Tinit
        Carnot_eff_L[cidx][0,:] = 1 - Toa[0] / construction.Tinit
        Carnot_eff_R[cidx][0,:] = 1 - Toa[0] / construction.Tinit

    # Convective heat transfer coefficients
    h_ci = 4
    R_ci = 1 / h_ci

    # Main simulation loop
    for n in tqdm(range(tN), desc="Simulation progress"):
        # Calculate heat gain and update indoor air temperature
        heat_gain = sum(h_ci * construction.area * (T_R[cidx][n, -1] - Tia[n,0]) for cidx ,construction in enumerate(structure)) * dt # [W]
        indoor_air.temp_update(heat_gain, Toa[n+1,0], simulation_time_parameters.dt)
        Tia[n+1,0] = indoor_air.temperature

        for cidx, construction in enumerate(structure):
            # Calculate outdoor convection coefficient
            h_co = cv.simple_combined_convection(construction.roughness, Vz)
            R_co = 1 / h_co

            # TDMA calculation
            T[cidx][n+1, :] = TDMA(construction, T[cidx][n, :], T_L[cidx][n, 0], T_R[cidx][n, -1], dt)

            # Calculate interface temperatures
            T_L[cidx][n+1, 0]  = (T[cidx][n+1, 0] / construction.R_L()[0]   + Toa[n+1,0] / R_co[n+1] + Normal_rad[n+1, 0]) / (1 / construction.R_L()[0] + 1 / R_co[n+1])
            T_R[cidx][n+1, -1] = (T[cidx][n+1, -1] / construction.R_R()[-1] + Tia[n+1,0] / R_ci) / (1 / R_ci + 1 / construction.R_R()[-1])
            
            # Linear interpolation for interface temperatures
            T_L[cidx][n+1, 1:]  = T[cidx][n+1, :-1] + (T[cidx][n+1, 1:] - T[cidx][n+1, :-1]) * (construction.dx[:-1] / (construction.dx[:-1] + construction.dx[1:]))
            T_R[cidx][n+1, :-1] = T[cidx][n+1, :-1] + (T[cidx][n+1, 1:] - T[cidx][n+1, :-1]) * (construction.dx[:-1] / (construction.dx[:-1] + construction.dx[1:]))
            
            # Calculate heat fluxes
            q_in[cidx][n+1, 1:] = construction.K_L[1:] * (T[cidx][n+1, :-1] - T[cidx][n+1, 1:])
            q_in[cidx][n+1, 0] = construction.K_L[0] * (T_L[cidx][n+1, 0] - T[cidx][n+1, 0])
            q_out[cidx][n+1, :-1] = construction.K_R[:-1] * (T[cidx][n+1, :-1] - T[cidx][n+1, 1:])
            q_out[cidx][n+1, -1] = construction.K_R[-1] * (T[cidx][n+1, -1] - T_R[cidx][n+1, -1])
            q[cidx][n+1, :] = (q_in[cidx][n+1, :] + q_out[cidx][n+1, :]) / 2

            # Calculate Carnot efficiency
            Carnot_eff_L[cidx][n+1, :] = 1 - Toa[n+1,0] / T_L[cidx][n+1, :]
            Carnot_eff_R[cidx][n+1, :] = 1 - Toa[n+1,0] / T_R[cidx][n+1, :]

    
    # Post-processing
    Tia_EPSD = eliminate_pre_simulation_data(Tia, simulation_time_parameters.ts_PST)
    Tia_EPSD_hf = half_time_matrix(Tia_EPSD, axis=0)

    for i, construction in enumerate(structure):
        # Eliminate pre-simulation data (EPSD)
        Toa_EPSD = eliminate_pre_simulation_data(Toa, simulation_time_parameters.ts_PST)
        T_L_EPSD = eliminate_pre_simulation_data(T_L[i], simulation_time_parameters.ts_PST)
        T_EPSD = eliminate_pre_simulation_data(T[i], simulation_time_parameters.ts_PST)
        T_R_EPSD = eliminate_pre_simulation_data(T_R[i], simulation_time_parameters.ts_PST)
        q_in_EPSD = eliminate_pre_simulation_data(q_in[i], simulation_time_parameters.ts_PST)
        q_EPSD = eliminate_pre_simulation_data(q[i], simulation_time_parameters.ts_PST)
        q_out_EPSD = eliminate_pre_simulation_data(q_out[i], simulation_time_parameters.ts_PST)
        Carnot_eff_L_EPSD = eliminate_pre_simulation_data(Carnot_eff_L[i], simulation_time_parameters.ts_PST)
        Carnot_eff_R_EPSD = eliminate_pre_simulation_data(Carnot_eff_R[i], simulation_time_parameters.ts_PST)

        # Calculate half time step values
        Toa_hf = half_time_matrix(Toa_EPSD, axis=0)
        T_L_hf = half_time_matrix(T_L_EPSD, axis=0)
        T_hf = half_time_matrix(T_EPSD, axis=0)
        T_R_hf = half_time_matrix(T_R_EPSD, axis=0)
        q_in_hf = half_time_matrix(q_in_EPSD, axis=0)
        q_hf = half_time_matrix(q_EPSD, axis=0)
        q_out_hf = half_time_matrix(q_out_EPSD, axis=0)
        Carnot_eff_L_hf = half_time_matrix(Carnot_eff_L_EPSD, axis=0)
        Carnot_eff_R_hf = half_time_matrix(Carnot_eff_R_EPSD, axis=0)

        # Data concatenation
        T_concat = np.concatenate((T_L_hf[:, 0].reshape(-1, 1), T_hf, T_R_hf[:, -1].reshape(-1, 1)), axis=1)
        q_node = q_hf
        q_intf = np.concatenate((q_in_hf, q_out_hf[:, -1].reshape(-1, 1)), axis=1)
        CXcR = (1 / construction.K()) * (Toa_hf * (q_hf / T_hf)**2)
        Carnot_eff_hf = np.concatenate((Carnot_eff_L_hf, Carnot_eff_R_hf[:, -1].reshape(-1, 1)), axis=1)

        # Create DataFrames
        x_node = [f"{round(i, 1)} cm" for i in np.cumsum(construction.dx_L()*c.m2cm)]
        x_node_BC = ['0.0 cm'] + x_node + [f"{construction.thick*c.m2cm:.1f} cm"]
        x_interface = ['0.0 cm'] + [f"{round(i, 1)} cm" for i in np.cumsum(construction.dx*c.m2cm)]

        T_df = pd.DataFrame(K2C(T_concat), columns=x_node_BC, index=time_index)
        q_node_df = pd.DataFrame(q_node, columns=x_node, index=time_index)
        q_intf_df = pd.DataFrame(q_intf, columns=x_interface, index=time_index)
        CXcR_df = pd.DataFrame(CXcR, columns=x_node, index=time_index)
        Carnot_eff_df = pd.DataFrame(Carnot_eff_hf, columns=x_interface, index=time_index)

        # Export to Excel
        filename = f"{construction.name}.xlsx"
        with pd.ExcelWriter(f"{filepath}/{filename}") as writer:
            T_df.to_excel(writer, sheet_name='T')
            q_node_df.to_excel(writer, sheet_name='q_node')
            q_intf_df.to_excel(writer, sheet_name='q_intf')
            CXcR_df.to_excel(writer, sheet_name='XcR')
            Carnot_eff_df.to_excel(writer, sheet_name='Carnot_eff')
    
    # Export indoor air temperature
    filename = "IndoorAir.xlsx"
    Tia_df = pd.DataFrame(K2C(Tia_EPSD_hf), columns=["Indoor Air Temperature [°C]"], index=time_index)
    Tia_df.to_excel(f"{filepath}/{filename}")

