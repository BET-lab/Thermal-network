# Thermal-network

This files attribute the function and class for 1 dimensional thermal network analysis

## Installation
```
pip install git+https://github.com/BET-lab/Thermal-network.git
```

## Usage

# Thermal Network Library

## 개요

이 라이브러리는 건물 열  네트워크 모델링 및 시뮬레이션을 위한 다양한 모듈을 제공합니다. 각 모듈은 특정한 역할을 수행하며, 상호 연결되어 열 전달, 복사, 대류, 기상 데이터 등의 처리를 지원합니다.

## 모듈 설명

### 1. `core.py`

- **기능:**
    - 단위 변환 (온도 변환: 섭씨 <-> 켈빈 등)
    - 데이터 처리 (numpy 배열과 pandas DataFrame 변환)
    - 시간 간격 보간 및 계산
    - 건물 내부 공기 모델링 (`IndoorAir` 클래스)
    - 건물 구조 및 층별 특성 모델링 (`Layer`, `Construction` 클래스)
    - TDMA(삼대각 행렬 알고리즘)를 활용한 열 전달 시뮬레이션 수행
    - 건물 엑서지 모델 실행

### 2. `location.py`

- **기능:**
    - 한국 내 주요 지역의 위도 및 경도 정보를 제공하는 딕셔너리
    - 특정 지역의 기상 데이터를 활용할 때 사용됨

### 3. `radiation.py`

- **기능:**
    - 태양 복사 관련 계산 수행
    - 태양 위치 (고도 및 방위각) 계산
    - 전천일사(GHI)를 직접일사(BHI) 및 확산일사(DHI)로 변환
    - 벽면 경사 및 방위각을 고려한 태양 복사량 계산
    - `get_solar_position()`, `get_ext_rad()`, `Erbs_diffuse_fraction()`, `solar_to_unit_surface()` 등의 주요 함수 포함

### 4. `weather.py`

- **기능:**
    - 기상 데이터 처리 및 보간 (입력값을 무조건 한 시간 간격의 데이터로 받음)
    - `SimulationTimeParameters` 클래스: 시뮬레이션 시간 설정 및 처리
    - `WeatherData` 클래스: 기온, 풍속, 일사량 데이터 전처리 및 변환
    - 태양의 위치(고도, 방위각) 계산 후 저장

### 5. `constant.py`

- **기능:**
    - 시간, 길이, 질량, 에너지, 압력 등의 단위 변환 상수 제공
    - 각종 기본 물리 상수 정의 (예: `h2s = 3600`, `J2kWh = 1/3.6e6` 등)

### 6. `convection.py`

- **기능:**
    - 외벽의 대류 열전달 계수 계산
    - 표면 거칠기에 따라 대류 열전달 계수를 결정하는 `simple_combined_convection()` 함수 포함

## 사용 예시

각 모듈은 독립적으로 사용될 수 있으며, `core.py`에서 주요 함수들을 호출하여 통합적인 시뮬레이션을 수행할 수 있습니다. 예를 들어:

```python
from thermal_network import core, weather, radiation

# 날씨 데이터 처리
sim_params = weather.SimulationTimeParameters(PST=1.0, MST=24.0, dt=3600)
weather_data = weather.WeatherData(location='서울', sim_time_params=sim_params,
                                   temperature=np.array([10, 15, 20, 25, 30]),
                                   wind_speed=np.array([1, 2, 3, 4, 5]),
                                   GHI_MJm2=np.array([0, 1, 2, 3, 4]))

# 태양 복사량 계산
solar_radiation = radiation.solar_to_unit_surface(weather_data, some_construction)

```

## 개선 필요 사항

- `core.py`:
    - 입력 온도를 켈빈 단위로 자동 변환 기능 추가 필요
- `radiation.py`:
    - 동/서쪽 방위각 처리 오류 수정 필요
- `weather.py`:
    - `start_time` 설정을 통한 유연한 시뮬레이션 시작 기능 추가 필요
    - 입력 기상 데이터의 시간간격을 자유롭게 설정할 수 있게 설정 필요 (현재는 1시간 간격이 디폴트임)
