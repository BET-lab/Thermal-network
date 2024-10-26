# import constant as c # absolute import (for module)

import os
import sys
from .import constant as c # relative import (for package)
from .import location as loc # relative import (for package)
import math
from datetime import datetime


### 1. Constants and Conversions

## 1.1 Angle Unit Conversions
d2r = math.pi/180
r2d = 180/math.pi
deg2min = 4 # 1 degree = 4 minutes
hour2deg = 15 # 1 hour = 15 degrees
deg_360 = 360
earth_axial_tilt = 23.45 # Earth's axial tilt [deg]


### 2. Solar Position Calculations

## 2.1 Equation of Time
def equation_of_time(day_of_year): 
    '''
    Calculates the Equation of Time: the difference between apparent solar time and mean solar time.
    Apparent solar time varies with the Earth's orbital motion, while mean solar time is averaged over the year.
    '''
    B = (day_of_year - 1) * 360/365 # B: angle in degrees

    EOT = 229.2 * (0.000075
               + 0.001868 * math.cos(d2r * B)
               - 0.032077 * math.sin(d2r * B)
               - 0.014615 * math.cos(d2r * 2 * B)
               - 0.04089 * math.sin(d2r * 2 * B)) 
    return EOT

## 2.2 Solar Position
def solar_position(station, year, month, day, local_hour, local_min, local_sec, standard_longitude = 135):
    '''
    Calculates the solar altitude and azimuth for a given location and time.
    '''

    local_latitude, local_longitude = loc.location[station]
    local_latitude, local_longitude = station
    
    # Equation of Time
    day_of_year = datetime(year, month, day).timetuple().tm_yday
    EOT = equation_of_time(day_of_year)
    
    # Solar Time
    delta_longitude = local_longitude - standard_longitude  # [deg]
    local_time_decimal = local_hour + local_min * c.m2h + local_sec * c.s2h  # [h]
    solar_time_decimal = local_time_decimal + (delta_longitude * deg2min + EOT) * c.m2h  # [h]
    
    hour_angle = solar_time_decimal * hour2deg - 180  # [deg]
    
    # Solar Declination
    solar_declination = earth_axial_tilt * math.sin(d2r * deg_360 * (284 + day_of_year) / c.y2d)
    
    # Solar Altitude
    term_1 = (math.cos(d2r * local_latitude) * math.cos(d2r * solar_declination) * math.cos(d2r * hour_angle)
              + math.sin(d2r * local_latitude) * math.sin(d2r * solar_declination))
    
    solar_altitude = r2d * math.asin(term_1)  # [deg]
    
    # Solar Azimuth
    term_2 = ((math.sin(d2r * solar_altitude) * math.sin(d2r * local_latitude) - math.sin(d2r * solar_declination))
              / (math.cos(d2r * solar_altitude) * math.cos(d2r * local_latitude)))
    
    if hour_angle < 0: # AM
        solar_azimuth = r2d * math.acos(term_2)
    elif hour_angle > 0: # PM
        solar_azimuth = 360 - r2d * math.acos(term_2)  # [deg]
    
    return solar_altitude, solar_azimuth


### 3. Solar Radiation Calculations

## 3.1 GHI to DNI and DHI Conversion
def GHI_MJm2_to_DNI_H_and_DHI(GHI_MJm2, station, year, month, day, local_hour):
    '''
    Converts Global Horizontal Irradiance (GHI) data [MJ/m2] to Direct Normal Irradiance on a Horizontal surface (DNI_H) [W/m2]
    and Diffuse Horizontal Irradiance (DHI) [W/m2] using an Excel-based calculation.
    '''
    # 현재 스크립트의 디렉토리 경로를 얻습니다
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Excel 파일의 경로를 구성합니다
    excel_path = os.path.join(current_dir, "DHI_calculator.xlsm")
    
    import xlwings as xw

    # Number of days in each month (ignoring leap years)
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Hide Excel application
    app = xw.App(visible=False)
    wb = None  # Initialize workbook as None
    try:
        wb = app.books.open(excel_path)  
        menu_sheet = wb.sheets["Main.Menu"]
        DHI_sheet = wb.sheets["DHI"]

        # Input data into Excel
        menu_sheet.range("C4").value = station
        menu_sheet.range("C8").value = year
        
        # Calculate cumulative hour (considering days in each month and hours)
        start_cell_number = 13
        cumulative_hour = start_cell_number + sum(days_in_month[:month]) * 24 + (day - 1) * 24 + local_hour

        # Input GHI values one by one
        for idx, value in enumerate(GHI_MJm2.values):
            DHI_sheet.range(f"H{cumulative_hour + idx}").value = value

        # Read DHI values
        GHI = DHI_sheet.range(f"J{cumulative_hour}:J{cumulative_hour + len(GHI_MJm2) - 1}").value  # [W/m2]
        DHI = DHI_sheet.range(f"K{cumulative_hour}:K{cumulative_hour + len(GHI_MJm2) - 1}").value  # [W/m2]

        # Calculate Direct Normal Irradiance on a Horizontal Surface
        DNI_H = [ghi - dhi for ghi, dhi in zip(GHI, DHI)]
        
        # Save Excel file
        wb.save()

    finally:
        # Close workbook and Excel application
        if wb:
            wb.close()
        app.quit()

    return DNI_H, DHI


## 3.2 Solar Radiation on Tilted Surface
def solar_to_unit_surface(DNI_H, DHI, station, year, month, day, local_hour, local_min, local_sec, surface_tilt, surface_azimuth):
    '''
    Calculates the total solar radiation on a tilted surface given the DNI_H, DHI, location, time, and surface orientation.
    '''
    VF = (1 + math.cos(d2r * surface_tilt))/2 # View factor between the sky and the surface
    DHI_surf = max(0, VF * DHI) # Sky Diffuse Radiation on a Tilted Surface [W/m2]

    # Solar Radiation to Surface
    solar_altitude, solar_azimuth = solar_position(station, year, month, day, local_hour, local_min, local_sec)
    delta_azimuth = abs(solar_azimuth - surface_azimuth) # [deg]

    # Calculate Direct Normal Irradiance on a Tilted Surface according to the surface azimuth cases
    if surface_azimuth >= 90:
        if surface_azimuth - 90 <= solar_azimuth <= surface_azimuth + 90:
            delta_azimuth = abs(solar_azimuth - surface_azimuth) # [deg]
            DNI_surf = max(0, math.sin(d2r * solar_altitude + d2r * surface_tilt) / math.sin(d2r * solar_altitude) * math.cos(d2r * delta_azimuth) * DNI_H) # [W/m2]
        else:
            DNI_surf = 0
    elif surface_azimuth < 90:
        if 0 <= solar_azimuth <= surface_azimuth + 90:
            delta_azimuth = abs(solar_azimuth - surface_azimuth) # [deg]
            DNI_surf = max(0, math.sin(d2r * solar_altitude + d2r * surface_tilt) / math.sin(d2r * solar_altitude) * math.cos(d2r * delta_azimuth) * DNI_H)
        elif solar_azimuth >= 360 - (90 - surface_azimuth):
            delta_azimuth = 360 - solar_azimuth + surface_azimuth
            DNI_surf = max(0, math.sin(d2r * solar_altitude + d2r * surface_tilt) / math.sin(d2r * solar_altitude) * math.cos(d2r * delta_azimuth) * DNI_H)
        else:
            DNI_surf = 0
        
    # When solar altitude is less than 5 degrees, the direct normal irradiance on a tilted surface is zero. (for error prevention)
    if solar_altitude < 5: # Solar Altitude < 5 deg
        DNI_surf = 0
    
    return max(0, DNI_surf + DHI_surf) # Total Solar Radiation on a Tilted Surface [W/m2]
