from thermal_network import constant as c

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
        0번째 단계와 마지막 타임스텝을 포함한 전체 시간 단계의 수
        
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
    start_time: pd.Timestamp = None  # Start time

    def __post_init__(self):
        # Calculate time steps
        self.TST = self.PST + self.MST  # Total-simulation time [hr]
        self.tN = int(1 + self.TST*c.h2s/self.dt) # Number of time steps +1 for the 0th time step
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

        if self.start_time is not None: 

            if not isinstance(self.start_time, pd.Timestamp):
                raise TypeError('start_time must be a pd.Timestamp')
            
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
            # start_time이 None인 경우의 기본 처리
            self.end_time = None
            self.time_step = None
            self.time_range = None
            self.time_range2 = None

    def get_pre_simulation_range(self):
        if self.time_range is None:
            raise ValueError("No time range available. Provide start_time during initialization.")
        return self.time_range[:self.ts_PST]

    def get_main_simulation_range(self):
        if self.time_range is None:
            raise ValueError("No time range available. Provide start_time during initialization.")
        return self.time_range[self.ts_PST:]

    def get_time_index(self, time):
        if self.time_range is None:
            raise ValueError("No time range available. Provide start_time during initialization.")
        return self.time_range.get_loc(time)

 
## 2.2 Weather Data Class
@dataclass
class WeatherData:
    '''
    24.12.16 수정 - solar altitude, azimuth의 중복 interpoation을 제거하였음
    get_solar_position 함수는 원래부터 interpolation을 수행하므로 중복 interpolation을 제거함
    
    기상청 데이터를 기반으로 쉽게 날씨데이터에 이용할 수 있게 전처리하는 클래스
    데이터의 길이를 시뮬레이션 시간간격 및 길이에 맞게 보간하고
    온도를 절대온도로 변환함
    태양의 고도 및 방위각을 계산하여 반환
    전천일사의 단위를 MJ/m²에서 W/m²로 변환함
    
    Parameters
    ----------
    location : str
        한국 내 지역명으로 "광주", "서울" 등과 같이 지역명을 한글 문자열로 입력
        
    sim_time_params : SimulationTimeParameters
        Weather data의 전처리 등을 위해 이용되는 클래스
        
    temperature: np.ndarray
        온도 데이터로 array
        
    wind_speed: np.ndarray
        풍속 데이터 array
        
    GHI_MJm2 : np.ndarray
        단위면적당 1시간동안 일사된 수평전천일사량으로 확산일사와 직달일사가 합산된 값임
        
    standard_longitude: float
        한국의 표준 경도로 기본값은 135로 설정
        
    Returns
    ------------------
    temp : np.array
        tN+1 타임스텝까지 절대온도로 변환된 온도 데이터 array
        
    wind_speed : np.array
        tN+1 타임스텝까지 보간된 풍속 데이터 array
        
    sol_alt : np.array
        tN+1 타임스텝까지 보간된 태양 고도 array
        
    sol_azi : np.array
        tN+1 타임스텝까지 보간된 태양 방위각 array
        
    GHI : np.array
        tN+1 타임스텝까지 보간된 일사량 데이터 array
        
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
    
    >>> # 기본 날씨 데이터 설정
    >>> weather = WeatherData(
    ...     location='서울',
    ...     sim_time_params=params,
    ...     temperature=np.array([10, 15, 20, 25, 30]),
    ...     wind_speed=np.array([1, 2, 3, 4, 5]),
    ...     GHI_MJm2=np.array([0, 1, 2, 3, 4])
    ... )
    
    '''
    location: str
    sim_time_params: SimulationTimeParameters
    temperature: np.ndarray
    wind_speed: np.ndarray
    GHI_MJm2: np.ndarray
    standard_longitude: float = 135  # standard longitude of Korea

    def __post_init__(self):
        from thermal_network import radiation as rd
        from thermal_network import core as core
        
        # temperature data       
        tN = self.sim_time_params.tN 
        temp_kelvin = core.C2K(self.temperature.copy())  # 섭씨->켈빈 변환
        temp_interpolated = core.interpolate_hourly_data(temp_kelvin, self.sim_time_params.dt)  # 보간
        self.temp = temp_interpolated[:tN+1]  # 0 step ~ last + 1 step 까지 추출 (시작은 전처리에서 처리해야함)
        
        # wind speed data
        self.wind_speed = core.interpolate_hourly_data(self.wind_speed, self.sim_time_params.dt)[:tN+1]
        
        # solar position data
        self.sol_alt, self.sol_azi = rd.get_solar_position(
            self.location, 
            self.sim_time_params, 
            self.standard_longitude
        )

        self.sol_alt = self.sol_alt[:tN+1]
        self.sol_azi = self.sol_azi[:tN+1]
        
        # Global Horizontal Irradiance (GHI) data (MJ/m² -> W/m²)
        self.GHI = core.interpolate_hourly_data(
                self.GHI_MJm2 * c.MJ2J / c.h2s, 
                self.sim_time_params.dt
            )[:tN+1]