from thermal_network import constant as c
from thermal_network import radiation as rd
from thermal_network import convection as cv
from thermal_network import weather as wd

# from .import constant as c
# from .import radiation as rd
# from .import convection as cv
# from .import weather as wd

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from dataclasses import dataclass

'''
필요 수정사항
1. Input 온도는 Kelvin 같은 단위 변환 자동화 시키기
2. excel 저장 -> csv 개별 파일 저장화하기
3. 태양복사가 외피 방위각과 상관없이 똑같이 적용되고 있음
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
def calculate_midpoint_matrix(mat: np.ndarray, axis: int) -> np.ndarray:
    """
    주어진 행렬의 인접한 행 또는 열 사이의 중간값을 계산합니다.

    이 함수는 입력 행렬의 연속된 두 요소 사이의 평균값을 계산하여 
    새로운 행렬을 반환합니다. axis 매개변수에 따라 행 방향 또는 
    열 방향으로 중간값을 계산할 수 있습니다.

    Parameters:
    -----------
    mat : np.ndarray
        입력 행렬
    axis : int
        중간값 계산 방향 
        - 0: 행(row) 방향으로 계산
        - 1: 열(column) 방향으로 계산

    Returns:
    --------
    np.ndarray
        중간값이 계산된 행렬

    Raises:
    -------
    ValueError
        axis가 0 또는 1이 아닌 경우 발생

    Examples:
    ---------
    >>> import numpy as np
    >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> get_half_time_matrix(mat, axis=0)
    array([[2.5, 3.5, 4.5],
           [5.5, 6.5, 7.5]])
    """
    
    if axis == 0:
        return (mat[:-1, :] + mat[1:, :]) / 2
    elif axis == 1:
        return (mat[:, :-1] + mat[:, 1:]) / 2
    else:
        raise ValueError("Axis must be 0 or 1")

## 1.5 Data Interpolation Function
def interpolate_hourly_data(hourly_data: np.ndarray, time_step: float) -> np.ndarray:
    """
    1시간 간격으로 얻어진 1차원 numpy 배열의 데이터를 
    주어진 time step 간격을 가진 새로운 배열로 보간하여 반환함
    
    Parameters:
    -----------
    hourly_data : np.ndarray
        단일 1시간 간격 1차원 numpy 배열
    time_step : float
        보간할 시간간격 [sec]

    Returns:
    --------
    np.ndarray
        보간된 데이터 배열

    Raises:
    -------
    ValueError


    Examples:
    ---------

    """
    
    rN = len(hourly_data)
    x = np.arange(rN)
    interval = time_step * c.s2h
    x_new = np.arange(0, rN + interval, interval)
    return np.interp(x_new, x, hourly_data)


### 2. Data Classes

## 2.3 Indoor Air Class
# @dataclass
# class IndoorAir:
#     ACH: float # Air change rate [1/hr]
#     h_ib: float # convective heat transfer coefficient [W/m2K]
#     volume: float # Volume of indoor air [m3]
#     specific_heat_capacity: float = 1005 # specific heat capacity of air [J/kgK]
#     density: float = 1.225 # density of air [kg/m3]
#     temperature: float = 20 # initial temperature [K]

#     def __post_init__(self):
#         self.C = self.specific_heat_capacity * self.density  # Volumetric heat capacity [J/m3K]
#         self.T = C2K(self.temperature) # Temperature [K]

#     def get_heat_capacity(self):
#         return self.volume * self.C # [J/K]
    
#     def volumetric_flow_rate(self):
#         return self.volume * self.ACH / c.h2s
    
#     def calcuate_thermal_energy_gain(self, Q_surf: float, Toa: float, dt: float):
#         Q_ACH = self.C * self.volumetric_flow_rate() * (Toa - self.T) * dt # [J]
#         return Q_surf + Q_ACH
    
#     def temp_update(self, heat_gain: float):
#         # Calculate indoor air temperature update
#         """heat_gain: [J]"""
#         dT = heat_gain / self.get_heat_capacity() # [K]
#         self.T += dT

@dataclass
class IndoorAir:
    temperature: float # [°C]
    volume: float # [m3]
    ACH: float # [-]
    h_ib: float # [W/m2K]
    specific_heat_capacity: float = 1005 # [J/kgK]
    density: float = 1.225 # [kg/m3]

    def __post_init__(self):
        self.C = self.specific_heat_capacity * self.density  # Volumetric heat capacity [J/m3K]
        self.temperature = C2K(self.temperature) # Temperature [K]
        
    def heat_gain_by_ACH(self, Toa: float, dt: float):
        return self.C * (self.volume * self.ACH / c.h2s) * dt * (Toa - self.temperature) # 체적열용량 * dt 동안 치환된 공기량 * 온도차 [J]
    
    def temp_update(self, heat_gain: float):
        # Calculate indoor air temperature update
        dT = heat_gain / (self.C * self.volume) # [K]
        self.temperature += dT


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
        self.discretization_num = round(self.L / self.dx)  # Number of discretization [-]
        
        # Thermal properties of the layer
        self.C = self.c * self.rho  # Volumetric heat capacity [J/m3K]
        self.R = self.dx / self.k  # Thermal resistance [m2K/W]
        self.K = self.k / self.dx  # Thermal conductance # [W/m2K]
        self.alpha = self.k / self.C  # Thermal diffusivity [m2/s]

## 2.5 Building Structure Class
@dataclass
class Construction:
    '''
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
        self.layer_discretization_counts_list = [layer.discretization_num for layer in self.layers]
        self.layer_num = len(self.layers)
        self.total_discretization_num = sum(self.layer_discretization_counts_list)
        self.thick = sum(layer.L for layer in self.layers)

        # Volumetric heat capacity [J/m3K]
        self.c = np.repeat([layer.c for layer in self.layers], self.layer_discretization_counts_list)
        self.C = np.repeat([layer.C for layer in self.layers], self.layer_discretization_counts_list)
        self.rho = np.repeat([layer.rho for layer in self.layers], self.layer_discretization_counts_list)

        # Cell size and distance between node [m]
        self.dx = np.repeat([layer.dx for layer in self.layers], self.layer_discretization_counts_list)
        self.dx_L = np.array([self.dx[0]/2] + [(self.dx[i-1] + self.dx[i])/2 for i in range(1, self.total_discretization_num)])
        self.dx_R = np.array([(self.dx[i] + self.dx[i+1])/2 for i in range(self.total_discretization_num-1)] + [self.dx[-1]/2])
        
        # Thermal resistance [m2K/W]
        self.R = np.repeat([layer.R for layer in self.layers], self.layer_discretization_counts_list)
        self.R_L = np.array([self.R[0]/2] + [(self.R[i-1] + self.R[i])/2 for i in range(1, self.total_discretization_num)])
        self.R_R = np.array([(self.R[i] + self.R[i+1])/2 for i in range(self.total_discretization_num-1)] + [self.R[-1]/2])
        
        # Themal conductance [W/m2K]
        self.K = 1 / self.R
        self.K_L = 1 / self.R_L
        self.K_R = 1 / self.R_R

        # Total thermal properties
        self.R_tot = np.sum(self.R)
        self.K_tot = 1 / self.R_tot

        # Calculate node center positions from left surface [m]
        self.node_pos_arr = np.array([np.cumsum(self.dx)[i] - self.dx[i]/2 for i in range(self.total_discretization_num)])*c.m2cm
        self.cell_surf_pos_arr = np.array([0] + [np.cumsum(self.dx)[i] for i in range(self.total_discretization_num)])*c.m2cm
        # self.node_pos_str_arr = [self.node_pos_arr[i]*c.m2cm.astype(str) + ' cm' for i in range(self.total_discretization_num)]
        # self.cell_surf_pos_str_arr = [self.cell_surf_pos_arr[i]*c.m2cm.astype(str) + ' cm' for i in range(self.total_discretization_num)]

    def get_heat_capacity_per_area(self):
        return np.sum(self.C * self.dx) # [J/m2K]

### 3. Simulation Functions

## 3.1 Data Processing Function
def extract_main_sim_data(data: np.ndarray, sim_time_params) -> np.ndarray:
    if data.ndim == 1:
        return data[sim_time_params.ts_PST:sim_time_params.tN]
    elif data.ndim == 2:
        return data[sim_time_params.ts_PST:sim_time_params.tN, :]

## 3.2 TDMA Calculation Function
def TDMA(Construction: Construction, T: np.ndarray, T_L: float, T_R: float, dt: float) -> np.ndarray:
    # TDMA (Tri-Diagonal Matrix Algorithm) calculation
    N = Construction.total_discretization_num
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


def run_building_exergy_model_fully_unsteady(
        structure : List[Construction],
        weather   : wd.WeatherData,
        indoor_air: IndoorAir,
        save_path : str,
        ): 
    
    # Parameters for simulation
    sim_time_params = weather.sim_time_params
    tN              = sim_time_params.tN # 0 time step ~ last time step
    dt              = sim_time_params.dt # time step [s]
    time_index      = sim_time_params.ts_h_main.astype(str)

    # Weather data(broadcasting)
    Toa = weather.temp.reshape(-1, 1) # outdoor air temperature [K] (reshape for broadcasting)
    GHI = weather.GHI.reshape(-1, 1) # Global Horizontal Irradiance [W/m2] (reshape for broadcasting)
    wind_speed = weather.wind_speed # Wind speed [m/s]
    
    # Overall heat transfer coefficients [W/m2K]
    h_ib = indoor_air.h_ib
    R_ib = 1 / h_ib
    
    # Initialize matrices # [component][time, node] 
    T                 = [np.zeros((tN+1, construction.total_discretization_num)) for construction in structure]
    T_L               = [np.zeros((tN+1, construction.total_discretization_num)) for construction in structure]
    T_R               = [np.zeros((tN+1, construction.total_discretization_num)) for construction in structure]
    q                 = [np.zeros((tN+1, construction.total_discretization_num)) for construction in structure]
    q_in              = [np.zeros((tN+1, construction.total_discretization_num)) for construction in structure]
    q_out             = [np.zeros((tN+1, construction.total_discretization_num)) for construction in structure]
    Tia               = np.full((tN+1,1), indoor_air.temperature) # indoor air temperature [K]
    q_ia              = np.zeros((tN + 1, 1)) # indoor air heat gain rate [W]
    Normal_irradiance = np.zeros((len(structure), tN+1)) # [W/m2]
    
    # Set initial conditions
    for cidx, construction in enumerate(structure):
        T[cidx][0,:] = construction.Tinit
        T_L[cidx][0,:] = construction.Tinit
        T_R[cidx][0,:] = construction.Tinit

    # Calculate solar radiation
    for i in range(len(structure)):
        Normal_irradiance[i] = structure[i].solar_absorptance * rd.solar_to_unit_surface(
                                            weather = weather,
                                            construction = structure[i],
                                            )

    # Main simulation time loop
    for n in tqdm(range(tN), desc="Simulation progress"):

        # Calculate heat gain and update indoor air temperature
        Q_surf = sum(h_ib * construction.area * (T_R[cidx][n, -1] - Tia[n,0]) for cidx, construction in enumerate(structure)) * dt # [J]
        Q_ACH = indoor_air.heat_gain_by_ACH(Toa[n,0], dt) # [J] 
        q_ia[n,0] = (Q_surf + Q_ACH) / dt # [W]
        indoor_air.temp_update(heat_gain = Q_surf + Q_ACH)
        Tia[n+1,0] = indoor_air.temperature
        
        # construction loop
        for cidx, construction in enumerate(structure):
            # Calculate outdoor convection coefficient
            h_co = cv.simple_combined_convection(construction.roughness, wind_speed) # [W/m2K]
            R_co = 1 / h_co

            # TDMA calculation
            T[cidx][n+1, :] = TDMA(construction, T[cidx][n, :], T_L[cidx][n, 0], T_R[cidx][n, -1], dt)

            # Calculate interface temperatures
            T_L[cidx][n+1, 0]  = (T[cidx][n+1, 0] / construction.R_L[0]   + Toa[n+1,0] / R_co[n+1] + Normal_irradiance[cidx][n+1]) / (1 / construction.R_L[0] + 1 / R_co[n+1])
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


    # n + 1/2 time step data 총 데이터 수 (tN+1 -> tN)
    Toa_hf = calculate_midpoint_matrix(Toa, axis=0)
    T_hf   = [calculate_midpoint_matrix(T[cidx], axis=0)   for cidx in range(len(structure))]
    T_L_hf = [calculate_midpoint_matrix(T_L[cidx], axis=0) for cidx in range(len(structure))]
    T_R_hf = [calculate_midpoint_matrix(T_R[cidx], axis=0) for cidx in range(len(structure))]

    q_hf     = [calculate_midpoint_matrix(q[cidx], axis=0)     for cidx in range(len(structure))]
    q_in_hf  = [calculate_midpoint_matrix(q_in[cidx], axis=0)  for cidx in range(len(structure))]
    q_out_hf = [calculate_midpoint_matrix(q_out[cidx], axis=0) for cidx in range(len(structure))]
    
    Carnot_eff_hf   = [1-Toa[:-1]/T_hf[cidx] for cidx in range(len(structure))] # 엑서지 저장률에 쓰이는 계수는 Toa는 그냥 n 타임스텝 쓰임 -> 마지막 값은 개수 맞추기 위해 제외 (tN+1 -> tN)
    Carnot_eff_L_hf = [1-Toa_hf/T_L_hf[cidx] for cidx in range(len(structure))] # 엑서지 인 아웃 플럭스에 쓰이는 계수는 Toa도 half time step
    Carnot_eff_R_hf = [1-Toa_hf/T_R_hf[cidx] for cidx in range(len(structure))] # 엑서지 인 아웃 플럭스에 쓰이는 계수는 Toa도 half time step
    
    
    # Post-processing (new axis for broadcasting)
    CXcR  = [construction.R[np.newaxis,:] * (Toa_hf * (q_hf[cidx] / T_hf[cidx])**2) for cidx, construction in enumerate(structure)] # Exergy consumption rate [W/m2]
    CXstR = [Carnot_eff_hf[cidx] * construction.dx[np.newaxis,:] * construction.C[np.newaxis,:] * (T[cidx][1:,:]-T[cidx][:-1,:])/dt for cidx, construction in enumerate(structure)] # Exergy storage rate [W/m2]
    CXst  = [construction.rho[np.newaxis,:] * construction.c[np.newaxis,:] * construction.dx[np.newaxis,:] * ((T[cidx]-Toa)-Toa*np.log(T[cidx]/Toa)) for cidx, construction in enumerate(structure)] # Stored exergy [J/m2]

    CXf_L         = [Carnot_eff_L_hf[cidx] * (q_in_hf[cidx]) for cidx,construction in enumerate(structure)] # Exergy flow [W/m2] 엑서지 플럭스의 경우는 Xout 의 맨 마지막을 제외하고는 인과 아웃이 일치한다.
    CXf_R         = [Carnot_eff_R_hf[cidx] * (q_out_hf[cidx]) for cidx,construction in enumerate(structure)] # Exergy flow [W/m2] 엑서지 플럭스의 경우는 Xout 의 맨 마지막을 제외하고는 인과 아웃이 일치한다.
    CXf           = [np.concatenate((CXf_L[cidx], CXf_R[cidx][:,-1:]), axis=1) for cidx, construction in enumerate(structure)]
    room_X_demand = ((1 - Toa/Tia) * q_ia) # [W]


    # 모든 데이터 pre simulation data 제거
    dict_DPSD1 = {}
    dict_DPSD2 = {}
    var1       = [T, q, Carnot_eff_hf, CXcR, Normal_irradiance, CXstR, CXst, CXf] # structure에 연관된 변수들
    var2       = [q_ia, room_X_demand, Tia, Toa, GHI] # structure에 독립적인 변수들
    var1_DPSD  = ['T_DPSD', 'q_DPSD', 'Carnot_eff_hf_DPSD', 'CXcR_DPSD','Normal_irradiance_DPSD', 'CXstR_DPSD', 'CXst_DPSD', 'CXf_DPSD'] # 결과를 저장할 변수명 설정
    var2_DPSD  = ['q_ia_DPSD', 'room_X_demand_DPSD', 'Tia_DPSD', 'Toa_DPSD', 'GHI_DPSD']

    # 구조체별 데이터 처리
    for var, result_name in zip(var1, var1_DPSD): 
        dict_DPSD1[result_name] = np.array([extract_main_sim_data(var[i], sim_time_params) for i in range(len(structure))])
    
    for var, result_name in zip(var2,var2_DPSD): 
        dict_DPSD2[result_name] = extract_main_sim_data(var, sim_time_params)

    ## DataFrame 생성 및 CSV 저장 ------------------------------------------------

    # Structure에 연관된 데이터 저장을 위한 설정
    data1 = {
        'T'                : (K2C(dict_DPSD1['T_DPSD']), construction.node_pos_arr.astype(str), '_T'),
        'q'                : (dict_DPSD1['q_DPSD'], construction.node_pos_arr.astype(str), '_q'),
        'XcR'              : (dict_DPSD1['CXcR_DPSD'], construction.node_pos_arr.astype(str), '_XcR'),
        'Carnot_eff'       : (dict_DPSD1['Carnot_eff_hf_DPSD'], construction.node_pos_arr.astype(str), '_Carnot_eff'),
        'CXstR'            : (dict_DPSD1['CXstR_DPSD'], construction.node_pos_arr.astype(str), '_CXstR'),
        'CXst'             : (dict_DPSD1['CXst_DPSD'], construction.node_pos_arr.astype(str), '_CXst'),
        'CXf'              : (dict_DPSD1['CXf_DPSD'], construction.cell_surf_pos_arr.astype(str), '_CXf'),
        'Normal_irradiance': (dict_DPSD1['Normal_irradiance_DPSD'], None, '_Normal_irradiance')
    }
    
    # Structure에 독립적인 데이터 저장을 위한 설정
    data2 = {
        'q_ia'         : (dict_DPSD2['q_ia_DPSD'],          ["Room Heat Gain [W]"],                 'Q_ia'),
        'room_X_demand': (dict_DPSD2['room_X_demand_DPSD'], ["Room Exergy Demand [W]"],             'room_exergy_demand'),
        'indoor_air'   : (K2C(dict_DPSD2['Tia_DPSD']),      ["Indoor Air Temperature [°C]"],        'indoor_air_temperature'),
        'outdoor_air'  : (K2C(dict_DPSD2['Toa_DPSD']),      ["Outdoor Air Temperature [°C]"],        'outdoor_air_temperature'),
        'ghi'          : (dict_DPSD2['GHI_DPSD'],           ["Global Horizontal Irradiance [W/m2]"], 'global_horizontal_irradiance')
    }

    def save_dataframe(data, columns, filename, index=time_index, save_path=save_path):
        """데이터프레임을 생성하고 CSV로 저장하는 헬퍼 함수"""
        df = pd.DataFrame(data, columns=columns, index=index)
        df.to_csv(f"{save_path}/{filename}.csv")
        return df
    
    # 구조체별 데이터 저장
    for _, (data, pos_arr_type, suffix) in data1.items():
        for cidx, construction in enumerate(structure):
            save_dataframe(data[cidx], pos_arr_type, f"{construction.name}{suffix}")

    # 단일 데이터 저장
    for _, (data, columns, filename) in data2.items():
        save_dataframe(data, columns, filename)

