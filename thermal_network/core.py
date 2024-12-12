# import constant as c
# import radiation as rd
# import convection as cv
# import weather as wd

from .import constant as c
from .import radiation as rd
from .import convection as cv
from .import weather as wd

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from dataclasses import dataclass

'''
필요 수정사항
1. Input 온도는 Kelvin 같은 단위 변환 자동화 시키기
2. excel 저장 -> csv 개별 파일 저장화하기
'''


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

## 2.3 Indoor Air Class
@dataclass
class IndoorAir:
    ACH: float # Air change rate [1/hr]
    h_ib: float # convective heat transfer coefficient [W/m2K]
    volume: float # Volume of indoor air [m3]
    specific_heat_capacity: float = 1005 # specific heat capacity of air [J/kgK]
    density: float = 1.225 # density of air [kg/m3]
    temperature: float = 20 # initial temperature [K]

    def __post_init__(self):
        self.C = self.specific_heat_capacity * self.density  # Volumetric heat capacity [J/m3K]
        self.T = C2K(self.temperature) # Temperature [K]

    def get_thermal_capacity(self):
        return self.volume * self.C # [J/K]
    
    def volumetric_flow_rate(self):
        return self.volume * self.ACH / c.h2s
    
    def get_heat_gain(self, Q_surf: float, Toa: float, dt: float):
        Q_ACH = self.C * self.volumetric_flow_rate() * (Toa - self.T) * dt # [J]
        return Q_surf + Q_ACH
    
    def temp_update(self, heat_gain: float):
        # Calculate indoor air temperature update
        dT = heat_gain / self.get_thermal_capacity() # [K]
        self.T += dT


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
    solar_absorptance: float
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
        
        # Convert initial temperature to Kelvin
        self.Tinit = C2K(self.Tinit)
        
        # Calculate physical properties of the structure
        self.layer_discr_counts = [layer.discr_num for layer in self.layers]
        self.layer_num = len(self.layers)
        self.total_discr_num = sum(self.layer_discr_counts)
        self.thick = sum(layer.L for layer in self.layers)

        # Volumetric heat capacity [J/m3K]
        self.c = np.repeat([layer.c for layer in self.layers], self.layer_discr_counts)
        self.C = np.repeat([layer.C for layer in self.layers], self.layer_discr_counts)
        self.rho = np.repeat([layer.rho for layer in self.layers], self.layer_discr_counts)

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

        # Calculate node center positions from left surface [m]
        cumsum_dx = np.cumsum(self.dx)
        self.node_positions = np.array([dx/2 for dx in self.dx])  # Initialize with half of first dx
        for i in range(1, len(self.dx)):
            self.node_positions[i] = cumsum_dx[i-1] + self.dx[i]/2

        # Create node center position strings in centimeters
        self.node_cent_pos = [f"{pos*c.m2cm:.1f} cm" for pos in self.node_positions]

        # Calculate surface positions from left surface [m]
        self.surface_positions = np.zeros(self.total_discr_num + 1)
        self.surface_positions[1:] = np.cumsum(self.dx)
        
        # Create surface position strings in centimeters
        self.node_surf_pos = [f"{pos*c.m2cm:.1f} cm" for pos in self.surface_positions]

    def get_thermal_capacity_per_area(self):
        return np.sum(self.C * self.dx) # [J/m2K]


### 3. Simulation Functions

## 3.1 Data Processing Function
def extract_main_sim_data(data: np.ndarray, sim_time_params) -> np.ndarray:
    if data.ndim == 1:
        return data[sim_time_params.ts_PST:sim_time_params.tN+1]
    elif data.ndim == 2:
        return data[sim_time_params.ts_PST:sim_time_params.tN+1, :]

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
    

## 3.4 Building exergy model simulation functions
def simulate_one_node_building_exergy(structure: List[Construction], sim_time_params: wd.SimulationTimeParameters,
                                                 weather: wd.WeatherData, indoor_air: IndoorAir, filepath: str):
    '''
    수정사항
    1. Xlsx 파일 -> 각 csv 파일로 저장
    '''

    # Simulation variables
    cN = len(structure) # compoenent number [-]
    dt = sim_time_params.dt # time step [s]
    tN = sim_time_params.tN # time step number [-]

    envelope_azi_arr = np.array([construction.azimuth for construction in structure])
    envelope_tilt_arr = np.array([construction.tilt for construction in structure])

    # Set outdoor and indoor conditions
    Toa = weather.temp.reshape(-1, 1) # outdoor air temperature [K] (reshape for broadcasting)
    Tia = np.full((tN,1), IndoorAir.temperature) # indoor air temperature [K]
    GHI = weather.GHI.reshape(-1, 1) # Global Horizontal Irradiance [W/m2] (reshape for broadcasting)

    # Calculate solar radiation
    Norm_rad = np.zeros((len(envelope_azi_arr), tN)) # [W/m2]
    for i, (azi, tilt) in enumerate(zip(envelope_azi_arr, envelope_tilt_arr)):
        Norm_rad[i] = rd.solar_to_unit_surface(
                                                BHI = weather.BHI, 
                                                DHI = weather.DHI, 
                                                station = weather.location, 
                                                SimulationTimeParameters = sim_time_params,
                                                surface_tilt=tilt, 
                                                surface_azimuth=azi
                                                )

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
    h_ib = indoor_air.h_ib
    R_ib = 1 / h_ib

    # Simulation time loop
    for n in tqdm(range(tN-1), desc="Simulation progress"):

        # Calculate heat gain and update indoor air temperature
        Q_surf = sum(h_ib * construction.area * (T[cidx][n,-1] - Tia[n,0]) for cidx ,construction in enumerate(structure)) * dt # heat gain by the component [W]
        indoor_air.temp_update(Q_surf, Toa[n+1,0], sim_time_params.dt)
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
            T[cidx][n+1, 0] = (T[cidx][n+1,1] / R_half + Toa[n+1,0] / R_co[n+1] + Norm_rad[cidx][n+1]) / (1 / R_half + 1 / R_co[n+1]) # left surface
            T[cidx][n+1,-1] = (T[cidx][n+1,1] / R_half + Tia[n+1,0] / R_ib) / (1 / R_ib + 1 / R_half) # right surface
            
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
        # Eliminate pre-simulation data (DPSD)
        T_DPSD = extract_main_sim_data(T[i], sim_time_params.ts_PST)
        q_DPSD = extract_main_sim_data(q[i], sim_time_params.ts_PST)
        Carnot_eff_DPSD = extract_main_sim_data(Carnot_eff[i], sim_time_params.ts_PST)
        CXcR_DPSD = extract_main_sim_data(CXcR[i], sim_time_params.ts_PST)
        Norm_rad_DPSD = extract_main_sim_data(Norm_rad[i], sim_time_params.ts_PST)

        # Data framing
        columns = ["OS", "Middle", "IS"]
        time_index = sim_time_params.time_range[sim_time_params.ts_PST:].astype(str)

        # Create DataFrames
        T_df = pd.DataFrame(K2C(T_DPSD), columns=columns, index=time_index)
        q_df = pd.DataFrame(q_DPSD, columns=columns, index=time_index)
        CXcR_df = pd.DataFrame(CXcR_DPSD, columns=None, index=time_index)
        Carnot_eff_df = pd.DataFrame(Carnot_eff_DPSD, columns=columns, index=time_index)
        Norm_rad_df = pd.DataFrame(Norm_rad_DPSD, columns=["Solar Radiation [W/m2]"], index=time_index)

        # Export to separate CSV files
        base_name = construction.name
        T_df.to_csv(f"{save_path}/{base_name}_T.csv")
        q_df.to_csv(f"{save_path}/{base_name}_q.csv")
        CXcR_df.to_csv(f"{save_path}/{base_name}_XcR.csv")
        Carnot_eff_df.to_csv(f"{save_path}/{base_name}_Carnot_eff.csv")
        Norm_rad_df.to_csv(f"{save_path}/{base_name}_Norm_rad.csv")

    # Post-processing and separate environment data files
    Tia_DPSD = extract_main_sim_data(Tia, sim_time_params.ts_PST)
    Toa_DPSD = extract_main_sim_data(Toa, sim_time_params.ts_PST)
    GHI_DPSD = extract_main_sim_data(GHI, sim_time_params.ts_PST)

    # Create and export indoor air temperature
    indoor_air_df = pd.DataFrame(
        K2C(Tia_DPSD), 
        columns=["Indoor Air Temperature [°C]"], 
        index=time_index
    )
    indoor_air_df.to_csv(f"{save_path}/indoor_air_temperature.csv")

    # Create and export outdoor air temperature
    outdoor_air_df = pd.DataFrame(
        K2C(Toa_DPSD), 
        columns=["Outdoor Air Temperature [°C]"], 
        index=time_index
    )
    outdoor_air_df.to_csv(f"{save_path}/outdoor_air_temperature.csv")

    # Create and export global horizontal irradiance
    ghi_df = pd.DataFrame(
        GHI_DPSD, 
        columns=["Global Horizontal Irradiance [W/m2]"], 
        index=time_index
    )
    ghi_df.to_csv(f"{save_path}/global_horizontal_irradiance.csv")
    

def run_building_exergy_model_fully_unsteady(
        structure: List[Construction],
        weather: wd.WeatherData,
        indoor_air: IndoorAir,
        save_path: str,
        ):
    ''' 미완성 '''
    # Main simulation function
    
    # Initialize simulation
    num_constructions = len(structure)
    sim_time_params = weather.sim_time_params
    tN = sim_time_params.tN
    dt = sim_time_params.dt
    time_index = sim_time_params.ts_h_main.astype(str)

    # Extract building structure parameters

    # Set outdoor and indoor conditions
    Toa = weather.temp.reshape(-1, 1) # outdoor air temperature [K] (reshape for broadcasting)
    Tia = np.full((tN+1,1), IndoorAir.temperature) # indoor air temperature [K]
    GHI = weather.GHI.reshape(-1, 1) # Global Horizontal Irradiance [W/m2] (reshape for broadcasting)
    Vz = weather.Vz

    # Calculate solar radiation
    sol_absorptance = np.array([construction.solar_absorptance for construction in structure])
    Norm_rad = np.zeros((num_constructions, tN+1)) # [W/m2]
    for i in range(num_constructions):
        Norm_rad[i] = sol_absorptance[i] * rd.solar_to_unit_surface(
                                            weather = weather,
                                            construction = structure[i],
                                            )

    # Initialize matrices # [component][time, node]
    node_nums = np.array([construction.total_discr_num for construction in structure]) # structure를 구성하는 construction 별 노드 개수를 따옴
    T = [np.zeros((tN+1, node_num)) for node_num in node_nums] 
    T_L = [np.zeros((tN+1, node_num)) for node_num in node_nums]
    T_R = [np.zeros((tN+1, node_num)) for node_num in node_nums]
    q = [np.zeros((tN+1, node_num)) for node_num in node_nums]
    q_in = [np.zeros((tN+1, node_num)) for node_num in node_nums]
    q_out = [np.zeros((tN+1, node_num)) for node_num in node_nums]
    

    # Set initial conditions
    for cidx, construction in enumerate(structure):
        T[cidx][0,:] = construction.Tinit
        T_L[cidx][0,:] = construction.Tinit
        T_R[cidx][0,:] = construction.Tinit

    # Overall heat transfer coefficients [W/m2K]
    h_ib = 7
    R_ib = 1 / h_ib

    # Main simulation time loop
    Q_room_gain = np.zeros((tN+1,1))
    for n in tqdm(range(tN), desc="Simulation progress"):

        # Calculate heat gain and update indoor air temperature
        Q_surf = sum(h_ib * construction.area * (T_R[cidx][n, -1] - Tia[n,0]) for cidx, construction in enumerate(structure)) * dt # [W]
        room_heat_gain = indoor_air.get_heat_gain(Q_surf, Toa[n+1,0], dt) # [J]
        Q_room_gain[n+1,0] = room_heat_gain/dt # [W]
        indoor_air.temp_update(heat_gain=room_heat_gain)
        Tia[n+1,0] = indoor_air.temperature
        
        # construction loop
        for cidx, construction in enumerate(structure):
            # Calculate outdoor convection coefficient
            h_co = cv.simple_combined_convection(construction.roughness, Vz) # [W/m2K]
            R_co = 1 / h_co

            # TDMA calculation
            T[cidx][n+1, :] = TDMA(construction, T[cidx][n, :], T_L[cidx][n, 0], T_R[cidx][n, -1], dt)

            # Calculate interface temperatures
            T_L[cidx][n+1, 0]  = (T[cidx][n+1, 0] / construction.R_L[0]   + Toa[n+1,0] / R_co[n+1] + Norm_rad[cidx][n+1]) / (1 / construction.R_L[0] + 1 / R_co[n+1])
            T_R[cidx][n+1, -1] = (T[cidx][n+1, -1] / construction.R_R[-1] + Tia[n+1,0] / R_ib) / (1 / R_ib + 1 / construction.R_R[-1])
            
            # Linear interpolation for interface temperatures
            T_L[cidx][n+1, 1:]  = T[cidx][n+1, :-1] + (T[cidx][n+1, 1:] - T[cidx][n+1, :-1]) * (construction.dx[:-1] / (construction.dx[:-1] + construction.dx[1:]))
            T_R[cidx][n+1, :-1] = T[cidx][n+1, :-1] + (T[cidx][n+1, 1:] - T[cidx][n+1, :-1]) * (construction.dx[:-1] / (construction.dx[:-1] + construction.dx[1:]))
            
            # Calculate heat fluxes
            q_in[cidx][n+1, 1:] = construction.K_L[1:] * (T[cidx][n+1, :-1] - T[cidx][n+1, 1:])
            q_in[cidx][n+1, 0] = construction.K_L[0] * (T_L[cidx][n+1, 0] - T[cidx][n+1, 0])
            q_out[cidx][n+1, :-1] = construction.K_R[:-1] * (T[cidx][n+1, :-1] - T[cidx][n+1, 1:])
            q_out[cidx][n+1, -1] = construction.K_R[-1] * (T[cidx][n+1, -1] - T_R[cidx][n+1, -1])
            q[cidx][n+1, :] = (q_in[cidx][n+1, :] + q_out[cidx][n+1, :]) / 2


    # half time matrix
    '''
    각 structure를 구성하는 construction 별로 half time matrix를 구함
    '''
    Toa_hf = half_time_matrix(Toa, axis=0)
    T_hf = [half_time_matrix(T[cidx], axis=0) for cidx, construction in enumerate(structure)]
    T_L_hf = [half_time_matrix(T_L[cidx], axis=0) for cidx, construction in enumerate(structure)]
    T_R_hf = [half_time_matrix(T_R[cidx], axis=0) for cidx, construction in enumerate(structure)]

    q_hf = [half_time_matrix(q[cidx], axis=0) for cidx, construction in enumerate(structure)]
    q_in_hf = [half_time_matrix(q_in[cidx], axis=0) for cidx, construction in enumerate(structure)]
    q_out_hf = [half_time_matrix(q_out[cidx], axis=0) for cidx, construction in enumerate(structure)]
    
    Carnot_eff_hf   = [1-Toa[:-1]/T_hf[cidx] for cidx, construction in enumerate(structure)] # 엑서지 저장률에 쓰이는 계수는 Toa는 그냥 쓰임 -> 마지막 값은 제외
    Carnot_eff_L_hf = [1-Toa_hf/T_L_hf[cidx] for cidx, construction in enumerate(structure)] # 엑서지 인 아웃 플럭스에 쓰이는 계수는 Toa도 half time step
    Carnot_eff_R_hf = [1-Toa_hf/T_R_hf[cidx] for cidx, construction in enumerate(structure)] # 엑서지 인 아웃 플럭스에 쓰이는 계수는 Toa도 half time step
    
    # Post-processing (new axis for broadcasting)
    CXcR = [construction.R[np.newaxis,:] * (Toa_hf * (q_hf[cidx] / T_hf[cidx])**2) for cidx, construction in enumerate(structure)] # Exergy consumption rate [W/m2]
    CXstR = [Carnot_eff_hf[cidx] * construction.dx[np.newaxis,:] * construction.C[np.newaxis,:] * (T[cidx][1:,:]-T[cidx][:-1,:])/dt for cidx, construction in enumerate(structure)] # Exergy storage rate [W/m2]
    CXst = [construction.rho[np.newaxis,:] * construction.c[np.newaxis,:] * construction.dx[np.newaxis,:] * ((T[cidx]-Toa)-Toa*np.log(T[cidx]/Toa)) for cidx, construction in enumerate(structure)] # Stored exergy [J/m2]

    CXf_L = [Carnot_eff_L_hf[cidx] * (q_in_hf[cidx]) for cidx,construction in enumerate(structure)] # Exergy flow [W/m2] 엑서지 플럭스의 경우는 Xout 의 맨 마지막을 제외하고는 인과 아웃이 일치한다. 
    CXf_R = [Carnot_eff_R_hf[cidx] * (q_out_hf[cidx]) for cidx,construction in enumerate(structure)] # Exergy flow [W/m2] 엑서지 플럭스의 경우는 Xout 의 맨 마지막을 제외하고는 인과 아웃이 일치한다.
    CXf = [np.concatenate((CXf_L[cidx], CXf_R[cidx][:,-1:]), axis=1) for cidx, construction in enumerate(structure)]
    room_X_demand = ((1 - Toa/Tia) * Q_room_gain) # [W]

    # 모든 데이터 pre simulation data 제거
    Q_room_gain_DPSD = extract_main_sim_data(Q_room_gain, sim_time_params)
    room_X_demand_DPSD = extract_main_sim_data(room_X_demand, sim_time_params)
    T_DPSD = [extract_main_sim_data(T[i], sim_time_params) for i in range(len(structure))]
    q_DPSD = [extract_main_sim_data(q[i], sim_time_params) for i in range(len(structure))]
    Carnot_eff_hf_DPSD = [extract_main_sim_data(Carnot_eff_hf[i], sim_time_params) for i in range(len(structure))]
    CXcR_DPSD = [extract_main_sim_data(CXcR[i], sim_time_params) for i in range(len(structure))]
    Norm_rad_DPSD = [extract_main_sim_data(Norm_rad[i], sim_time_params) for i in range(len(structure))]
    CXstR_DPSD = [extract_main_sim_data(CXstR[i], sim_time_params) for i in range(len(structure))]
    CXst_DPSD = [extract_main_sim_data(CXst[i], sim_time_params) for i in range(len(structure))]
    CXf_DPSD = [extract_main_sim_data(CXf[i], sim_time_params) for i in range(len(structure))]

    # DataFrame 생성
    Q_room_gain_df = pd.DataFrame(Q_room_gain_DPSD, columns=["Room Heat Gain [W]"], index=time_index)
    room_X_demand_df = pd.DataFrame(room_X_demand_DPSD, columns=["Room Exergy Demand [W]"], index=time_index)
    T_dfs = [pd.DataFrame(K2C(T_DPSD[cidx]), columns=construction.node_cent_pos, index=time_index) for cidx, construction in enumerate(structure)]
    q_dfs = [pd.DataFrame(q_DPSD[cidx], columns=construction.node_cent_pos, index=time_index) for cidx, construction in enumerate(structure)]
    CXcR_dfs = [pd.DataFrame(CXcR_DPSD[cidx], columns=construction.node_cent_pos, index=time_index) for cidx, construction in enumerate(structure)]
    Carnot_eff_dfs = [pd.DataFrame(Carnot_eff_hf_DPSD[cidx], columns=construction.node_cent_pos, index=time_index) for cidx, construction in enumerate(structure)]
    Norm_rad_dfs = [pd.DataFrame(Norm_rad_DPSD[cidx], columns=["Solar Radiation [W/m2]"], index=time_index) for cidx, construction in enumerate(structure)]
    CXstR_dfs = [pd.DataFrame(CXstR_DPSD[cidx], columns=construction.node_cent_pos, index=time_index) for cidx, construction in enumerate(structure)]
    CXst_dfs = [pd.DataFrame(CXst_DPSD[cidx], columns=construction.node_cent_pos, index=time_index) for cidx, construction in enumerate(structure)]
    CXf_dfs = [pd.DataFrame(CXf_DPSD[cidx], columns=construction.node_surf_pos, index=time_index) for cidx, construction in enumerate(structure)]

    # CSV 파일 저장
    [T_dfs[i].to_csv(f"{save_path}/{structure[i].name}_T.csv") for i in range(len(structure))]
    [q_dfs[i].to_csv(f"{save_path}/{structure[i].name}_q.csv") for i in range(len(structure))]
    [CXcR_dfs[i].to_csv(f"{save_path}/{structure[i].name}_XcR.csv") for i in range(len(structure))]
    [Carnot_eff_dfs[i].to_csv(f"{save_path}/{structure[i].name}_Carnot_eff.csv") for i in range(len(structure))]
    [Norm_rad_dfs[i].to_csv(f"{save_path}/{structure[i].name}_Norm_rad.csv") for i in range(len(structure))]
    [CXstR_dfs[i].to_csv(f"{save_path}/{structure[i].name}_CXstR.csv") for i in range(len(structure))]
    [CXst_dfs[i].to_csv(f"{save_path}/{structure[i].name}_CXst.csv") for i in range(len(structure))]
    [CXf_dfs[i].to_csv(f"{save_path}/{structure[i].name}_CXf.csv") for i in range(len(structure))]

    # 환경 데이터 처리
    Tia_DPSD = extract_main_sim_data(Tia, sim_time_params)
    Toa_DPSD = extract_main_sim_data(Toa, sim_time_params)
    GHI_DPSD = extract_main_sim_data(GHI, sim_time_params)

    # Post-processing and separate environment data files
    Tia_DPSD = extract_main_sim_data(Tia, sim_time_params)
    Toa_DPSD = extract_main_sim_data(Toa, sim_time_params)
    GHI_DPSD = extract_main_sim_data(GHI, sim_time_params)

    # Create and export indoor air temperature
    Q_room_gain_df.to_csv(
        f"{save_path}/room_heat_gain.csv",
                )
    
    room_X_demand_df.to_csv(
        f"{save_path}/room_exergy_demand.csv",
        )

    indoor_air_df = pd.DataFrame(
        K2C(Tia_DPSD), 
        columns=["Indoor Air Temperature [°C]"], 
        index=time_index
    )

    # Create and export outdoor air temperature
    outdoor_air_df = pd.DataFrame(
        K2C(Toa_DPSD),
        columns=["Outdoor Air Temperature [°C]"], 
        index=time_index
    )

    # Create and export global horizontal irradiance
    ghi_df = pd.DataFrame(
        GHI_DPSD, 
        columns=["Global Horizontal Irradiance [W/m2]"], 
        index=time_index
    )

    indoor_air_df.to_csv(f"{save_path}/indoor_air_temperature.csv")
    outdoor_air_df.to_csv(f"{save_path}/outdoor_air_temperature.csv")
    ghi_df.to_csv(f"{save_path}/global_horizontal_irradiance.csv")
