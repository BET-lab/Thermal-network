import numpy as np
import pandas as pd
from . import constant as c
from . import radiation as rd
from . import convection as cv
from tqdm import tqdm
from typing import List
from dataclasses import dataclass

# Temperature conversion functions
def C2K(temp_C: float) -> float:
    return temp_C + 273.15

def K2C(temp_K: float) -> float:
    return temp_K - 273.15

def F2C(temp_F: float) -> float:
    return (temp_F - 32) * 5 / 9

def C2F(temp_C: float) -> float:
    return temp_C * 9 / 5 + 32

def cm2in(cm: float) -> float:
    return cm / 2.54



# Array and DataFrame conversion functions
def arr2df(arr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(arr)

def df2arr(df: pd.DataFrame) -> np.ndarray:
    return df.values



# Time interval calculation functions
def half_time_vector(vec: np.ndarray) -> np.ndarray:
    return (vec[:-1] + vec[1:]) / 2

def half_time_matrix(mat: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        return (mat[:-1, :] + mat[1:, :]) / 2
    elif axis == 1:
        return (mat[:, :-1] + mat[:, 1:]) / 2
    else:
        raise ValueError("Axis must be 0 or 1")

# Data interpolation function
def interpolate_hourly_data(hourly_data: np.ndarray, time_step: float) -> np.ndarray:
    rN = len(hourly_data)
    x = np.arange(rN)
    interval = time_step * c.s2h
    x_new = np.arange(0, rN + interval, interval)
    return np.interp(x_new, x, hourly_data)



# Simulation time related class
@dataclass
class SimulationTime:
    PST: float  # Pre-simulation time [hr]
    MST: float  # Main-simulation time [hr]
    dt: float   # Time step [s]

    def __post_init__(self):
        # Calculate simulation times
        self.TST = self.PST + self.MST  # Total-simulation time [hr]
        self.tN = int((self.TST) * c.h2s / self.dt) + 1  # Total number of time steps +1 for 0.0 (0.0 - TST [h])
        self.ts_PST = int(self.PST * c.h2s / self.dt)  # Number of pre-simulation time steps
        self.ts_MST = int(self.MST * c.h2s / self.dt)  # Number of main-simulation time steps
        self.ts_TST = int(self.TST * c.h2s / self.dt)  # Number of total-simulation time steps
        self.ts_s = np.arange(0, self.tN * self.dt, self.dt)
        self.ts_m = self.ts_s * c.s2m
        self.ts_h = self.ts_s * c.s2h

# Weather data related class
@dataclass
class Weather:
    '''
    Gh는 수평 전천일사를 의미함
    '''
    year: int
    month: int
    day: int
    local_hour: int
    local_min: int
    local_sec: int
    local_latitude: float
    local_longitude: float
    standard_longitude: float
    temp: np.ndarray
    Vz: np.ndarray
    Gh: np.ndarray



# Indoor air related class
@dataclass
class IndoorAir:
    temperature: float # [K]
    volume: float # [m3]
    ACH: float # [-]
    specific_heat_capacity: float = 1005 # [J/kgK]
    density: float = 1.225 # [kg/m3]

    def __post_init__(self):
        self.C = self.specific_heat_capacity * self.density  # Volumetric heat capacity [J/m3K]

    def temp_update(self, heat_gain: float, Toa: float, dt: float):
        # Calculate indoor air temperature update
        dT_by_ACH = self.C * (self.volume * self.ACH / c.h2s) * (Toa - self.temperature) * dt /(self.volume * self.C) # [K]
        dT_by_heat_gain = heat_gain / (self.volume * self.C) # [K]
        self.temperature += dT_by_ACH + dT_by_heat_gain

# Building material layer related class
@dataclass
class SetLayer:
    L: float  # Length [m]
    dx: float  # dx [m]
    k: float  # Thermal conductivity [W/mK]
    c: float  # Specific heat capacity [J/kgK]
    rho: float  # Density [kg/m3]

    def __post_init__(self):
        # Calculate physical properties of the layer
        self.div = round(self.L / self.dx)  # Number of division [-]
        self.C = self.c * self.rho  # Volumetric heat capacity [J/m^3K]
        self.R = self.dx / self.k  # Thermal resistance [m^2K/W]
        self.K = self.k / self.dx  # Thermal conductance # [W/m^2K]
        self.alpha = self.k / self.C  # Thermal diffusivity [m^2/s]

# Building structure related class
@dataclass
class SetConstruction:
    name: str
    layers: List[SetLayer]
    area: float
    roughness: str
    azimuth: float
    tilt: float
    Tinit: float

    def __post_init__(self):
        # Calculate physical properties of the structure
        self.div_list = [layer.div for layer in self.layers]
        self.lN = len(self.layers)
        self.N = sum(self.div_list)
        self.tick = sum(layer.L for layer in self.layers)

        roughness_list = ['very rough', 'rough', 'medium rough', 'medium smooth', 'smooth', 'very smooth']
        if self.roughness not in roughness_list:
            raise ValueError(f"roughness must be one of {roughness_list}")

    # Methods to calculate thermal properties of the structure
    def dx(self) -> np.ndarray:
        return np.repeat([layer.dx for layer in self.layers], self.div_list)

    def dx_L(self) -> np.ndarray:
        dx = self.dx()
        return np.array([dx[0]/2] + [(dx[i-1] + dx[i])/2 for i in range(1, self.N)])

    def dx_R(self) -> np.ndarray:
        dx = self.dx()
        return np.array([(dx[i] + dx[i+1])/2 for i in range(self.N-1)] + [dx[-1]/2])

    def R(self) -> np.ndarray:
        return np.repeat([layer.R for layer in self.layers], self.div_list)

    def K(self) -> np.ndarray:
        return 1 / self.R()

    def K_L(self) -> np.ndarray:
        R = self.R()
        return 1 / np.array([R[0]/2] + [(R[i-1] + R[i])/2 for i in range(1, self.N)])

    def K_R(self) -> np.ndarray:
        R = self.R()
        return 1 / np.array([(R[i] + R[i+1])/2 for i in range(self.N-1)] + [R[-1]/2])

    def R_L(self) -> np.ndarray:
        R = self.R()
        return np.array([R[0]/2] + [(R[i-1] + R[i])/2 for i in range(1, self.N)])

    def R_R(self) -> np.ndarray:
        R = self.R()
        return np.array([(R[i] + R[i+1])/2 for i in range(self.N-1)] + [R[-1]/2])

    def K_tot(self) -> float:
        return 1 / self.R_tot()

    def R_tot(self) -> float:
        return np.sum(self.R())

    def C(self) -> np.ndarray:
        return np.repeat([layer.C for layer in self.layers], self.div_list)
@dataclass
class SetSingleCapacitanceComponent:
    name: str
    roughness: str
    L: float  # Length [m]
    k: float  # Thermal conductivity [W/mK]
    c: float  # Specific heat capacity [J/kgK]
    rho: float  # Density [kg/m3]
    area: float  # Area [m2]
    azimuth: float
    tilt: float
    Tinit: float

    def __post_init__(self):
        self.C = self.c * self.rho
        self.R = self.L / self.k
        self.K = self.k / self.L
        self.V = self.area * self.L



# Simulation related functions
def eliminate_pre_simulation_data(data: np.ndarray, pre_simulation_time_step: int) -> np.ndarray:
    if data.ndim == 1:
        return data[pre_simulation_time_step:]
    elif data.ndim == 2:
        return data[pre_simulation_time_step:, :]

def TDMA(Construction: SetConstruction, T: np.ndarray, T_L: float, T_R: float, dt: float) -> np.ndarray:
    # TDMA (Tri-Diagonal Matrix Algorithm) calculation
    N = Construction.N
    dx = Construction.dx()
    K_L, K_R = Construction.K_L(), Construction.K_R()
    C = Construction.C()
    
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

def calculate_solar_radiation(Weather: Weather, envelope_azi: np.ndarray, envelope_tilt: np.ndarray, tN: int, dt: float) -> np.ndarray:
    # Calculate solar radiation
    q_rad = np.zeros((tN, len(envelope_azi)))
    current_time = pd.Timestamp(Weather.year, Weather.month, Weather.day, 
                                Weather.local_hour, Weather.local_min, Weather.local_sec)
    
    for n in range(tN):
        sol_alt, sol_azi = rd.solar_position(current_time.year, current_time.month, current_time.day,
                                             current_time.hour, current_time.minute, current_time.second,
                                             Weather.local_latitude, Weather.local_longitude, Weather.standard_longitude)
        
        for i, (azi, tilt) in enumerate(zip(envelope_azi, envelope_tilt)):
            q_rad[n, i] = rd.solar_to_unit_surface(Weather.Gh[n], sol_azi, sol_alt, tilt, azi)
        
        current_time += pd.Timedelta(seconds=dt)
    
    return q_rad

def run_building_exergy_model_single_capacitance(Structure: List[SetSingleCapacitanceComponent], SimulationTime: SimulationTime,
                                                 Weather: Weather, IndoorAir: IndoorAir, filepath: str):
    num_components = len(Structure)
    tN = SimulationTime.tN
    dt = SimulationTime.dt
    time_index = SimulationTime.ts_h

    envelope_azi = np.array([component.azimuth for component in Structure])
    envelope_tilt = np.array([component.tilt for component in Structure])

    # Set outdoor and indoor conditions
    Toa = Weather.temp.reshape(-1, 1)
    Vz = Weather.Vz
    q_rad = calculate_solar_radiation(Weather, envelope_azi, envelope_tilt, tN, dt)
    Tia = np.full((tN,1), IndoorAir.temperature)

    # Initialize matrices
    node_num = 3 # 0: Left node, 1: Center node, 2: Right node
    T = np.zeros((num_components, tN, node_num)) # [component, time, node]
    q = np.zeros_like(T)
    Carnot_eff = np.zeros_like(T)
    

     # Set initial conditions
    for cidx, construction in enumerate(Structure):
        T[cidx][0,:] = construction.Tinit
        q[cidx][0,:] = 0
        Carnot_eff[cidx][0,:] = 1 - Toa[0] / construction.Tinit

    # Convective heat transfer coefficients
    h_ci = 4
    R_ci = 1 / h_ci

    # Main simulation loop
    for n in tqdm(range(tN-1), desc="Simulation progress"):
        # Calculate heat gain and update indoor air temperature
        heat_gain = sum(h_ci * construction.area * (T[cidx][n,-1] - Tia[n,0]) for cidx ,construction in enumerate(Structure)) * dt # [W]
        IndoorAir.temp_update(heat_gain, Toa[n+1,0], SimulationTime.dt)
        Tia[n+1,0] = IndoorAir.temperature

        for cidx, construction in enumerate(Structure):
            # Calculate outdoor convection coefficient
            h_co = cv.simple_combined_convection(construction.roughness, Vz)
            R_co = 1 / h_co

            # Calculate center temperature & heat flux
            T[cidx][n+1, 1] =  T[cidx][n,1] + (q[cidx][n,0] - q[cidx][n,-1]) * dt / (construction.C * construction.L)
           
            # Calculate interface temperatures & heat fluxes (explicit method)
            R_half= construction.R/2
            T[cidx][n+1, 0] = (T[cidx][n+1,1] / R_half + Toa[n+1,0] / R_co[n+1] + q_rad[n+1, 0]) / (1 / R_half + 1 / R_co[n+1]) # left surface
            T[cidx][n+1,-1] = (T[cidx][n+1,1] / R_half + Tia[n+1,0] / R_ci) / (1 / R_ci + 1 / R_half) # right surface
            
            q[cidx][n+1, 0] = (T[cidx][n+1,0] - T[cidx][n+1,1]) / (R_half)
            q[cidx][n+1,-1] = (T[cidx][n+1,1] - T[cidx][n+1,-1]) / (R_half)
            

            # Calculate heat fluxes
            q[cidx][n+1, 1] = (q[cidx][n+1,0] + q[cidx][n+1,-1]) / 2

            # Calculate Carnot efficiency
            Carnot_eff[cidx][n+1, :] = 1 - Toa[n+1,0] / T[cidx][n+1, :]

    CXcR = (construction.R) * (Toa * (q[:][:,1] / T[:][:,1])**2) # Exergy consumption rate [W/m3]


    # Post-processing
    Tia_EPSD = eliminate_pre_simulation_data(Tia, SimulationTime.ts_PST)

    for i, construction in enumerate(Structure):
        # Eliminate pre-simulation data (EPSD)
        Toa_EPSD = eliminate_pre_simulation_data(Toa, SimulationTime.ts_PST)
        T_EPSD = eliminate_pre_simulation_data(T[i], SimulationTime.ts_PST)
        q_EPSD = eliminate_pre_simulation_data(q[i], SimulationTime.ts_PST)
        Carnot_eff_EPSD = eliminate_pre_simulation_data(Carnot_eff[i], SimulationTime.ts_PST)
        CXcR_EPSD = eliminate_pre_simulation_data(CXcR, SimulationTime.ts_PST)

        columns = ["OS", "Middle", "IS"]
        T_df = pd.DataFrame(K2C(T_EPSD), columns=columns, index=time_index[SimulationTime.ts_PST:])
        q_df = pd.DataFrame(q_EPSD, columns=columns, index=time_index[SimulationTime.ts_PST:])
        CXcR_df = pd.DataFrame(CXcR_EPSD, columns=None, index=time_index[SimulationTime.ts_PST:])
        Carnot_eff_df = pd.DataFrame(Carnot_eff_EPSD, columns=columns, index=time_index[SimulationTime.ts_PST:])

        # Export to Excel
        filename = f"{construction.name}.xlsx"
        with pd.ExcelWriter(f"{filepath}/{filename}") as writer:
            T_df.to_excel(writer, sheet_name='T')
            q_df.to_excel(writer, sheet_name='q')
            CXcR_df.to_excel(writer, sheet_name='XcR')
            Carnot_eff_df.to_excel(writer, sheet_name='Carnot_eff')
    
    # Export indoor air temperature
    Tia_df = pd.DataFrame(K2C(Tia_EPSD), columns=["Indoor Air Temperature [°C]"], index=time_index[SimulationTime.ts_PST:])
    Toa_df = pd.DataFrame(K2C(Toa_EPSD), columns=["Outdoor Air Temperature [°C]"], index=time_index[SimulationTime.ts_PST:])
    Tia_df.to_excel(f"{filepath}/IndoorAir.xlsx")
    Toa_df.to_excel(f"{filepath}/OutdoorAir.xlsx")
    

def run_building_exergy_model_fully_unsteady(Structure: List[SetConstruction], SimulationTime: SimulationTime, 
                         Weather: Weather, IndoorAir: IndoorAir, filepath: str):
    # Main simulation function
    
    # Initialize simulation
    num_constructions = len(Structure)
    tN = SimulationTime.tN
    dt = SimulationTime.dt
    time_index = SimulationTime.ts_h

    # Extract building structure parameters
    envelope_azi = np.array([construction.azimuth for construction in Structure])
    envelope_tilt = np.array([construction.tilt for construction in Structure])

    # Set outdoor and indoor conditions
    Toa = Weather.temp.reshape(-1, 1)
    Vz = Weather.Vz
    q_rad = calculate_solar_radiation(Weather, envelope_azi, envelope_tilt, tN, dt)
    Tia = np.full((tN+1,1), IndoorAir.temperature)

    # Initialize matrices
    T = np.zeros((num_constructions, tN + 1, max(construction.N for construction in Structure)))
    T_L = np.zeros_like(T)
    T_R = np.zeros_like(T)
    q_in = np.zeros_like(T)
    q = np.zeros_like(T)
    q_out = np.zeros_like(T)
    Carnot_eff_L = np.zeros_like(T)
    Carnot_eff_R = np.zeros_like(T)



    # Set initial conditions
    for cidx, construction in enumerate(Structure):
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
        heat_gain = sum(h_ci * construction.area * (T_R[cidx][n, -1] - Tia[n,0]) for cidx ,construction in enumerate(Structure)) * dt # [W]
        IndoorAir.temp_update(heat_gain, Toa[n+1,0], SimulationTime.dt)
        Tia[n+1,0] = IndoorAir.temperature

        for cidx, construction in enumerate(Structure):
            # Calculate outdoor convection coefficient
            h_co = cv.simple_combined_convection(construction.roughness, Vz)
            R_co = 1 / h_co

            # TDMA calculation
            T[cidx][n+1, :] = TDMA(construction, T[cidx][n, :], T_L[cidx][n, 0], T_R[cidx][n, -1], dt)

            # Calculate interface temperatures
            T_L[cidx][n+1, 0]  = (T[cidx][n+1, 0] / construction.R_L()[0]   + Toa[n+1,0] / R_co[n+1] + q_rad[n+1, 0]) / (1 / construction.R_L()[0] + 1 / R_co[n+1])
            T_R[cidx][n+1, -1] = (T[cidx][n+1, -1] / construction.R_R()[-1] + Tia[n+1,0] / R_ci) / (1 / R_ci + 1 / construction.R_R()[-1])
            
            # Linear interpolation for interface temperatures
            T_L[cidx][n+1, 1:]  = T[cidx][n+1, :-1] + (T[cidx][n+1, 1:] - T[cidx][n+1, :-1]) * (construction.dx()[:-1] / (construction.dx()[:-1] + construction.dx()[1:]))
            T_R[cidx][n+1, :-1] = T[cidx][n+1, :-1] + (T[cidx][n+1, 1:] - T[cidx][n+1, :-1]) * (construction.dx()[:-1] / (construction.dx()[:-1] + construction.dx()[1:]))
            
            # Calculate heat fluxes
            q_in[cidx][n+1, 1:] = construction.K_L()[1:] * (T[cidx][n+1, :-1] - T[cidx][n+1, 1:])
            q_in[cidx][n+1, 0] = construction.K_L()[0] * (T_L[cidx][n+1, 0] - T[cidx][n+1, 0])
            q_out[cidx][n+1, :-1] = construction.K_R()[:-1] * (T[cidx][n+1, :-1] - T[cidx][n+1, 1:])
            q_out[cidx][n+1, -1] = construction.K_R()[-1] * (T[cidx][n+1, -1] - T_R[cidx][n+1, -1])
            q[cidx][n+1, :] = (q_in[cidx][n+1, :] + q_out[cidx][n+1, :]) / 2

            # Calculate Carnot efficiency
            Carnot_eff_L[cidx][n+1, :] = 1 - Toa[n+1,0] / T_L[cidx][n+1, :]
            Carnot_eff_R[cidx][n+1, :] = 1 - Toa[n+1,0] / T_R[cidx][n+1, :]

    
    # Post-processing
    Tia_EPSD = eliminate_pre_simulation_data(Tia, SimulationTime.ts_PST)
    Tia_EPSD_hf = half_time_matrix(Tia_EPSD, axis=0)

    for i, construction in enumerate(Structure):
        # Eliminate pre-simulation data (EPSD)
        Toa_EPSD = eliminate_pre_simulation_data(Toa, SimulationTime.ts_PST)
        T_L_EPSD = eliminate_pre_simulation_data(T_L[i], SimulationTime.ts_PST)
        T_EPSD = eliminate_pre_simulation_data(T[i], SimulationTime.ts_PST)
        T_R_EPSD = eliminate_pre_simulation_data(T_R[i], SimulationTime.ts_PST)
        q_in_EPSD = eliminate_pre_simulation_data(q_in[i], SimulationTime.ts_PST)
        q_EPSD = eliminate_pre_simulation_data(q[i], SimulationTime.ts_PST)
        q_out_EPSD = eliminate_pre_simulation_data(q_out[i], SimulationTime.ts_PST)
        Carnot_eff_L_EPSD = eliminate_pre_simulation_data(Carnot_eff_L[i], SimulationTime.ts_PST)
        Carnot_eff_R_EPSD = eliminate_pre_simulation_data(Carnot_eff_R[i], SimulationTime.ts_PST)

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
        x_node_BC = ['0.0 cm'] + x_node + [f"{construction.tick*c.m2cm:.1f} cm"]
        x_interface = ['0.0 cm'] + [f"{round(i, 1)} cm" for i in np.cumsum(construction.dx()*c.m2cm)]

        T_df = pd.DataFrame(K2C(T_concat), columns=x_node_BC, index=time_index[SimulationTime.ts_PST:])
        q_node_df = pd.DataFrame(q_node, columns=x_node, index=time_index[SimulationTime.ts_PST:])
        q_intf_df = pd.DataFrame(q_intf, columns=x_interface, index=time_index[SimulationTime.ts_PST:])
        CXcR_df = pd.DataFrame(CXcR, columns=x_node, index=time_index[SimulationTime.ts_PST:])
        Carnot_eff_df = pd.DataFrame(Carnot_eff_hf, columns=x_interface, index=time_index[SimulationTime.ts_PST:])

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
    Tia_df = pd.DataFrame(K2C(Tia_EPSD_hf), columns=["Indoor Air Temperature [°C]"], index=time_index[SimulationTime.ts_PST:])
    Tia_df.to_excel(f"{filepath}/{filename}")
    



sim_time = SimulationTime(PST=0, MST=24.0, dt=10)

# Define weather data
weather_data = Weather(
    year=2023,
    month=1,
    day=1,
    local_hour=0,
    local_min=0,
    local_sec=0,
    local_latitude=37.7749,
    local_longitude=-122.4194,
    standard_longitude=-120,
    temp=np.random.uniform(low=C2K(0), high=C2K(20), size=sim_time.tN),
    Vz=np.random.uniform(low=0, high=5, size=sim_time.tN),
    Gh=np.random.uniform(low=0, high=800, size=sim_time.tN)
)

# Define indoor air conditions
indoor_air = IndoorAir(
    temperature=C2K(20),  # 20°C in Kelvin
    volume=100.0,  # 100 m3
    ACH=0.1  # Air changes per hour
)

# Define building structure
structure = [
    SetSingleCapacitanceComponent(
        name="Wall",
        roughness="medium rough",
        L=0.2,  # 20 cm
        k=0.5,  # W/mK
        c=1000,  # J/kgK
        rho=2200,  # kg/m3
        area=10.0,  # m2
        azimuth=180,  # South-facing
        tilt=90,  # Vertical
        Tinit=C2K(20)  # Initial temperature 20°C in Kelvin
    )
]

# Define file path for output
output_filepath = "../output"

# Run the building exergy model
run_building_exergy_model_single_capacitance(
    Structure=structure,
    SimulationTime=sim_time,
    Weather=weather_data,
    IndoorAir=indoor_air,
    filepath=output_filepath
)