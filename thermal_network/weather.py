from .import constant as c
# import constant as c

import numpy as np
import pandas as pd
from dataclasses import dataclass


## 2.1 Simulation Time Class
@dataclass
class SimulationTimeParameters:  # 필요 수정사항: stat_time 설정에 따라 weather data의 시작 시간을 설정할 수 있도록 수정
    '''
    시뮬레이션 시간 관련 매개변수를 관리하는 클래스
    
    주요 기능:
    - 시뮬레이션의 전체 시간 범위 설정 (PST + MST)
    - 시간 간격(dt)에 따른 시간 배열 생성
    - 선택적으로 실제 날짜/시간 기반의 시뮬레이션 시간 범위 설정
    
    Attributes
    ----------
    PST : float
        Pre-simulation time (사전 시뮬레이션 시간) [hr]
        시뮬레이션 시작 전 안정화를 위한 시간
        
    MST : float
        Main-simulation time (주 시뮬레이션 시간) [hr]
        실제 시뮬레이션이 수행되는 시간
        
    dt : float
        Time step (시간 간격) [s]
        시뮬레이션의 각 단계 사이의 시간 간격
        
    start_time : pd.Timestamp, optional
        Simulation start time (시뮬레이션 시작 시각)
        실제 날짜/시간 기준 시뮬레이션 시작 시각 (선택적)
        
    Generated Attributes
    ------------------
    TST : float
        Total simulation time (전체 시뮬레이션 시간) [hr]
        PST + MST
        
    tN : int
        Total number of time steps (전체 시간 단계 수)
        0번째 단계를 포함한 전체 시간 단계의 수
        
    ts_PST : int
        Number of pre-simulation time steps (사전 시뮬레이션 시간 단계 수)
        
    ts_MST : int
        Number of main-simulation time steps (주 시뮬레이션 시간 단계 수)
        
    ts_TST : int
        Total number of simulation time steps (전체 시간 단계 수)
        
    Time Arrays
    -----------
    ts_s_tot_dt : ndarray
        전체 시간 단계 배열 [s], 0번째 단계 포함
    ts_m_tot_dt : ndarray
        전체 시간 단계 배열 [min], 0번째 단계 포함
    ts_h_tot_dt : ndarray
        전체 시간 단계 배열 [hr], 0번째 단계 포함
        
    ts_s_tot : ndarray
        전체 시간 단계 배열 [s]
    ts_m_tot : ndarray
        전체 시간 단계 배열 [min]
    ts_h_tot : ndarray
        전체 시간 단계 배열 [hr]
        
    ts_s_main : ndarray
        주 시뮬레이션 시간 단계 배열 [s]
    ts_m_main : ndarray
        주 시뮬레이션 시간 단계 배열 [min]
    ts_h_main : ndarray
        주 시뮬레이션 시간 단계 배열 [hr]
        
    ts_s_pre : ndarray
        사전 시뮬레이션 시간 단계 배열 [s]
    ts_m_pre : ndarray
        사전 시뮬레이션 시간 단계 배열 [min]
    ts_h_pre : ndarray
        사전 시뮬레이션 시간 단계 배열 [hr]
        
    Timestamp-based Attributes (when start_time is provided)
    ----------------------------------------------------
    end_time : pd.Timestamp
        시뮬레이션 종료 시각
    time_step : pd.Timedelta
        시간 간격
    time_range : pd.DatetimeIndex
        시뮬레이션 전체 시간 범위
    time_range2 : pd.DatetimeIndex
        시뮬레이션 전체 시간 범위 (마지막 시간 단계 포함)
    
    Methods
    -------
    get_pre_simulation_range()
        사전 시뮬레이션 기간의 시간 범위 반환
    get_main_simulation_range()
        주 시뮬레이션 기간의 시간 범위 반환
    get_time_index(time)
        특정 시각의 인덱스 반환
        
    Examples
    --------
    >>> # 기본 시뮬레이션 시간 설정
    >>> params = SimulationTimeParameters(PST=1.0, MST=24.0, dt=3600)
    >>> 
    >>> # 실제 시각 기반 시뮬레이션 설정
    >>> params_with_time = SimulationTimeParameters(
    ...     PST=1.0,
    ...     MST=24.0,
    ...     dt=3600,
    ...     start_time=pd.Timestamp('2024-01-01')
    ... )
    '''
    PST: float  # Pre-simulation time [hr]
    MST: float  # Main-simulation time [hr]
    dt: float   # Time step [s]
    start_time = None  # Start time

    def __post_init__(self):
        # Calculate time steps
        self.TST = self.PST + self.MST  # Total-simulation time [hr]
        self.tN = int(self.TST*c.h2s/self.dt + 1) # Number of time steps +1 for the 0th time step
        self.ts_PST = int(self.PST * c.h2s / self.dt)  # Number of pre-simulation time steps
        self.ts_MST = int(self.MST * c.h2s / self.dt)  # Number of main-simulation time steps
        self.ts_TST = self.ts_PST + self.ts_MST  # Number of total-simulation time steps
        
        # Generate time arrays
        self.ts_s_tot_dt = np.arange(0, self.tN + 1) * self.dt # time step array [s]
        self.ts_m_tot_dt = self.ts_s_tot_dt * c.s2m # time step array [min]
        self.ts_h_tot_dt = self.ts_s_tot_dt * c.s2h # time step array [hr]
        
        self.ts_s_tot = np.arange(0, self.tN) * self.dt  # time step array [s]
        self.ts_m_tot = self.ts_s_tot * c.s2m  # time step array [min]
        self.ts_h_tot = self.ts_s_tot * c.s2h  # time step array [hr]

        self.ts_s_main = np.arange(self.ts_PST, self.tN) * self.dt  # time step array [s]
        self.ts_m_main = self.ts_s_main * c.s2m 
        self.ts_h_main = self.ts_s_main * c.s2h

        self.ts_s_pre = np.arange(0, self.ts_PST) * self.dt  # time step array [s]
        self.ts_m_pre = self.ts_s_pre * c.s2m
        self.ts_h_pre = self.ts_s_pre * c.s2h

        if self.start_time.dtype == pd.Timestamp:
            # Calculate simulation times
            self.end_time = self.start_time + pd.Timedelta(hours=self.TST)
            self.time_step = pd.Timedelta(seconds=self.dt)
            
            # Generate time range
            '''
            time_range: 시작 시간부터 종료 시간까지 시간 간격을 time_step으로 설정한 시간 배열
            '''
            # 1년 시작부터의 누적 시간 계산
            # accumulated_hours = (self.start_time.dayofyear - 1) * 24 + self.start_time.hour
            # self.start_idx = int(accumulated_hours * c.h2s / self.dt)  # 초 단위로 변환 후 dt로 나눔
            self.time_range = pd.date_range(start=self.start_time, end=self.end_time, freq = self.time_step)
            self.time_range2 = pd.date_range(start=self.start_time, end=self.end_time + pd.Timedelta(seconds=self.dt), freq=self.time_step)
        else: 
            pass

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
        from . import radiation as rd
        from . import core as core
        
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
        
