from .import constant as c
# import constant as c

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
    
    '''
    필요 수정사항: stat_time 설정에 따라 weather data의 시작 시간을 설정할 수 있도록 수정
    '''

    def __post_init__(self):
        # Calculate simulation times
        self.TST = self.PST + self.MST  # Total-simulation time [hr]
        self.end_time = self.start_time + pd.Timedelta(hours=self.TST)
        self.time_step = pd.Timedelta(seconds=self.dt)
        
        # Generate time range
        '''
        time_range: 시작 시간부터 종료 시간까지 시간 간격을 time_step으로 설정한 시간 배열
        '''
        # 1년 시작부터의 누적 시간 계산
        # accumulated_hours = (self.start_time.dayofyear - 1) * 24 + self.start_time.hour
        # self.start_idx = int(accumulated_hours * c.h2s / self.dt)  # 초 단위로 변환 후 dt로 나눔
        self.time_range = pd.date_range(start=self.start_time, end=self.end_time, freq=self.time_step)
        self.time_range2 = pd.date_range(start=self.start_time, end=self.end_time + pd.Timedelta(seconds=self.dt), freq=self.time_step)
        
        # Calculate time steps
        self.tN = len(self.time_range)  # Number of time steps
        self.ts_PST = int(self.PST * c.h2s / self.dt)  # Number of pre-simulation time steps
        self.ts_MST = int(self.MST * c.h2s / self.dt)  # Number of main-simulation time steps
        self.ts_TST = self.ts_PST + self.ts_MST  # Number of total-simulation time steps
        
        # Generate time arrays
        self.ts_s_total = np.arange(0, self.tN) * self.dt  # time step array [s]
        self.ts_m_total = self.ts_s_total * c.s2m  # time step array [min]
        self.ts_h_total = self.ts_s_total * c.s2h  # time step array [hr]

        self.ts_s_main = np.arange(self.ts_PST, self.tN) * self.dt  # time step array [s]
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
    temperature: np.ndarray
    Vz: np.ndarray
    GHI_MJm2: np.ndarray
    standard_longitude: float = 135  # standard longitude of Korea

    def __post_init__(self):
        import radiation as rd
        import core
        
        # 원본 데이터 임시 저장
        temp_original = self.temperature.copy()  # 원본 데이터 보존

        # cut
        # start_idx = self.sim_time_params.start_idx
        # end_idx = start_idx + self.sim_time_params.tN + 1
        
        # Temperature 처리
        temp_kelvin = core.C2K(temp_original)  # 섭씨->켈빈 변환
        temp_interpolated = core.interpolate_hourly_data(temp_kelvin, self.sim_time_params.dt)  # 보간
        self.temp = temp_interpolated[:self.sim_time_params.tN+1]  # 추출
        
        # 나머지 데이터 처리
        self.Vz = core.interpolate_hourly_data(self.Vz, self.sim_time_params.dt)[:self.sim_time_params.tN+1]
        
        solar_position = rd.get_solar_position(
            self.location, 
            self.sim_time_params, 
            self.standard_longitude
        )[:self.sim_time_params.tN+1]

        self.sol_alt = core.interpolate_hourly_data(solar_position[0], self.sim_time_params.dt)[:self.sim_time_params.tN+1]

        self.sol_azi = core.interpolate_hourly_data(solar_position[1], self.sim_time_params.dt)[:self.sim_time_params.tN+1]
        
        self.GHI = core.interpolate_hourly_data(
                self.GHI_MJm2 * c.MJ2J / c.h2s, 
                self.sim_time_params.dt
            )[:self.sim_time_params.tN+1]
        