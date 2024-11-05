from .import constant as c
from .import radiation as rd

import numpy as np
import pandas as pd
from dataclasses import dataclass


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
        '''
        time_range: 시작 시간부터 종료 시간까지 시간 간격을 time_step으로 설정한 시간 배열
        '''
        self.time_range = pd.date_range(start=self.start_time, end=self.end_time, freq=self.time_step)
        
        # Calculate time steps
        self.tN = len(self.time_range)
        self.ts_PST = int(self.PST * c.h2s / self.dt)  # Number of pre-simulation time steps
        self.ts_MST = int(self.MST * c.h2s / self.dt)  # Number of main-simulation time steps
        self.ts_TST = self.ts_PST + self.ts_MST  # Number of total-simulation time steps
        
        # Generate time arrays
        self.ts_s_total = np.arange(0, self.tN) * self.dt  # time step array [s]
        self.ts_m_total = self.ts_s * c.s2m  # time step array [min]
        self.ts_h_total = self.ts_s * c.s2h  # time step array [hr]

        self.ts_s_main = np.arange(self.ts_PST, self.ts_TST) * self.dt  # time step array [s]
        self.ts_m_main = self.ts_s_main * c.s2m 
        self.ts_h_main = self.ts_s_main * c.s2h

        self.ts_s_pre = np.arange(0, self.ts_PST) * self.dt  # time step array [s]
        self.ts_m_pre = self.ts_s_pre * c.s2m
        self.ts_h_pre = self.ts_s_pre * c.s2h

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
    sim_time_params: SimulationTimeParameters
    temp: np.ndarray
    Vz: np.ndarray
    GHI_MJm2: np.ndarray
    standard_longitude: float = 135 # standard longitude of Korea

    def __post_init__(self):
        self.sol_alt, self.sol_azi = rd.get_solar_position(self.location, self.sim_time_params, self.standard_longitude)
        self.BHI, self.DHI = rd.get_BHI_DHI(self.GHI_MJm2, self.sol_alt, self.sim_time_params.time_range.dayofyear)
        self.GHI = self.GHI_MJm2 * c.MJ2J / c.h2s
        
