import math
from datetime import datetime
import constant as c


# 각도 단위 변경
d2r = math.pi/180
r2d = 180/math.pi
deg2min = 4 # 1도 = 4분
hour2deg = 15 # 1시간 = 15도
deg_360 = 360
earth_axial_tilt = 23.45 # 지구의 기울기 [deg]



def equation_of_time(day_of_year): 
    '''
    Equation of Time: 진태양시와 평균태양시의 차이 
    진태양시란 지구의 공전궤도에 따라 변하는 하루의 시간을 고려한 시간이며
    평균태양시란 이를 24시간으로 365일 평균화하여 계산하여 쓰고 있는 시간이다. 
    '''

    B = (day_of_year - 1) * 360/365 # B: 각도 단위

    EOT = 229.2 * (0.000075
               + 0.001868 * math.cos(d2r * B)
               - 0.032077 * math.sin(d2r * B)
               - 0.014615 * math.cos(d2r * 2 * B)
               - 0.04089 * math.sin(d2r * 2 * B)) 
    return EOT



def solar_position(year, month, day, local_hour, local_min, local_sec, local_latitude, local_longitude, standard_longitude):
    
    # Equation of Time
    day_of_year = datetime(year, month, day).timetuple().tm_yday
    EOT = equation_of_time(day_of_year)
    
    # Solar Time
    delta_longitude = local_longitude - standard_longitude  # [deg] # 경도 차이
    local_time_decimal = local_hour + local_min * c.m2h + local_sec * c.s2h  # [h] # 지방의 표준시간
    solar_time_decimal = local_time_decimal + (delta_longitude * deg2min + EOT) * c.m2h  # [h] 진태양시, 표준 경도와 지역 경도의 차이에 의한 시간차, 공전에 의한 균시차를 고려한 진짜 태양기반 시간

    
    hour_angle = solar_time_decimal * hour2deg - 180  # [deg]
    """
    hour_angle (시간각)
    시간각은 태양의 현재 위치와 남중 시각(태양이 정남쪽에 있는 시각) 사이의 각도 차이를 나타냄
    이는 지구의 자전에 따른 태양의 겉보기 운동을 측정하는 데 사용됩니다.
    """
    
    # Solar Declination
    solar_declination = earth_axial_tilt * math.sin(d2r * deg_360  * (284 + day_of_year) / c.y2d)
    """ 
    solar_declination (태양적위)
    태양의 중심과 지구의 중심을 연결하는 선이 지구의 적도면이 이루는 각도
    """
    
    # Solar Altitude, Solar Azimuth
    term_1 = (math.cos(d2r * local_latitude) * math.cos(d2r * solar_declination) * math.cos(d2r * hour_angle)
              + math.sin(d2r * local_latitude) * math.sin(d2r * solar_declination))
    
    solar_altitude = r2d * math.asin(term_1)  # [deg]
    
    term_2 = ((math.sin(d2r * solar_altitude) * math.sin(d2r * local_latitude) - math.sin(d2r * solar_declination))
              / (math.cos(d2r * solar_altitude) * math.cos(d2r * local_latitude)))
    
    solar_azimuth = r2d * math.acos(term_2)  # [deg]
    
    return solar_altitude, solar_azimuth


def solar_to_unit_surface(global_solar_radiation, solar_altitude, solar_azimuth, surface_tilt, surface_azimuth):

    # Solar Radiation to Surface
    delta_azimuth = surface_azimuth - solar_azimuth # [deg]
    sol_to_surface = math.sin(d2r * solar_altitude + d2r * surface_tilt) * math.cos(d2r * delta_azimuth) * global_solar_radiation # [W/m2]
    
    return sol_to_surface

