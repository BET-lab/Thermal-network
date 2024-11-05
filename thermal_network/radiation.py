# 1. Imports and Setup
from .import constant as c  # relative import (for package)
from .import location as loc  # relative import (for package)
from .import weather as wd  # relative import (for package)

import math
from datetime import datetime
import numpy as np

# 2. Constants and Conversions
## 2.1. Angle Unit Conversions
d2r = math.pi/180  # Degrees to radians
r2d = 180/math.pi  # Radians to degrees
deg2min = 4        # 1 degree = 4 minutes
hour2deg = 15      # 1 hour = 15 degrees
deg_360 = 360
earth_axial_tilt = 23.45  # Earth's axial tilt [deg]

# 3. Solar Position Functions
## 3.1. Equation of Time
def equation_of_time(day_of_year):
    """
    get the difference between true solar time and mean solar time.
    
    Args:
        day_of_year (int): Day of year (1-365)
    
    Returns:
        float: Equation of time in minutes
    """
    B = (day_of_year - 1) * 360/365  # B: angle in degrees

    EOT = 229.2 * (0.000075
               + 0.001868 * np.cos(d2r * B)
               - 0.032077 * np.sin(d2r * B)
               - 0.014615 * np.cos(d2r * 2 * B)
               - 0.04089 * np.sin(d2r * 2 * B)) 
    return EOT

## 3.2. Solar Position Calculator
def get_solar_position(station, SimulationTimeParameters, standard_longitude=135):
    """
    get solar altitude and azimuth for a given location and time.
    
    Args:
        station (str): Observatory name (e.g., 'Gwangju')
        SimulationTimeParameters (object): Time information object
        standard_longitude (float): Standard longitude (default: 135)
    
    Returns:
        tuple: (solar_altitude, solar_azimuth) in degrees
    """
    # Convert time information to numpy arrays
    local_hour = np.array(SimulationTimeParameters.time_range.hour)
    local_min = np.array(SimulationTimeParameters.time_range.minute)
    local_sec = np.array(SimulationTimeParameters.time_range.second)

    # Location information
    local_latitude, local_longitude = loc.location[station]
    
    # get day of year
    day_of_year = np.array(SimulationTimeParameters.time_range.dayofyear)
    
    # get equation of time
    EOT = equation_of_time(day_of_year)
    
    # get solar time
    delta_longitude = local_longitude - standard_longitude
    local_time_decimal = local_hour + local_min * c.m2h + local_sec * c.s2h
    solar_time_decimal = local_time_decimal + (delta_longitude * deg2min + EOT) * c.m2h
    
    # get hour angle (-180 to 180 degrees, negative for east, positive for west)
    hour_angle = solar_time_decimal * hour2deg - 180
    
    # get solar declination
    solar_declination = earth_axial_tilt * np.sin(d2r * deg_360 * (284 + day_of_year) / c.y2d)
    
    # get solar altitude
    term_1 = (np.cos(d2r * local_latitude) * np.cos(d2r * solar_declination) * np.cos(d2r * hour_angle)
              + np.sin(d2r * local_latitude) * np.sin(d2r * solar_declination))
    solar_altitude = r2d * np.arcsin(term_1)

    # get solar azimuth (0° at north, clockwise)
    term_2 = (np.sin(d2r * solar_declination) * np.cos(d2r * local_latitude) -
              np.cos(d2r * hour_angle) * np.cos(d2r * solar_declination) * np.sin(d2r * local_latitude)) / np.sin(d2r * (90-solar_altitude))
    
    solar_azimuth = np.where(np.sign(hour_angle) == -1,
                            r2d * np.arccos(term_2),
                            360 - r2d * np.arccos(term_2))

    return solar_altitude, solar_azimuth

# 4. Solar Radiation Functions
## 4.1. Extraterrestrial Radiation
def get_ext_rad(
        weather: wd.WeatherData,
                ):
    """
    get extraterrestrial radiation on a horizontal surface.
    
    Args:
        solar_altitude (float/array): Solar altitude angle in radians
        day_of_year (int/array): Day of year (1-365)
    
    Returns:
        float/array: Extraterrestrial radiation (W/m²)
    """
    sol_alt = weather.sol_alt
    day_of_year = weather.sim_time_params.time_range.dayofyear

    solar_constant = 1367  # W/m²
    distance_correction = 1 + 0.033 * np.cos(np.radians(360 * day_of_year / 365))
    I_0 = solar_constant * distance_correction * np.sin(sol_alt * d2r)
    return np.maximum(0, I_0)

## 4.2. Erbs Model for Diffuse Fraction
def Erbs_diffuse_fraction(
        GHI: np.array,
        solar_altitude: np.array,
        day_of_year: np.array,
        ):
    """
    get diffuse horizontal irradiance using Erbs et al. (1982) model.
    
    Args:
        GHI (array-like): Global Horizontal Irradiance (W/m²)
        solar_altitude (array-like): Solar altitude angle (radians)
        day_of_year (array-like): Day of year (1-365)
    
    Returns:
        tuple: (DHI)
    """
    GHI_arr = np.array(GHI)
    I0 = get_ext_rad(solar_altitude, day_of_year)
    kt = GHI_arr / I0
    
    diffuse_fraction = np.where(
        kt <= 0.22,
        1.0 - 0.09 * kt,
        np.where(
            kt <= 0.80,
            0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
            0.165
        )
    )
    
    DHI = GHI_arr * diffuse_fraction
    return DHI

## 4.3. Solar Radiation Conversion Functions
def get_BHI_DHI(
        weather: wd.WeatherData,
        ):
    """
    Convert Global Horizontal Irradiance to Beam and Diffuse components.
    
    Args:
        GHI_MJm2 (float/array): Global Horizontal Irradiance (MJ/m²)
        solar_altitude (float/array): Solar altitude angle
        day_of_year (int/array): Day of year
    
    Returns:
        tuple: (BHI, DHI) in W/m²
    """
    GHI = weather.GHI_MJm2 * c.MJ2J/c.h2s
    DHI = Erbs_diffuse_fraction(GHI, weather.sol_alt, weather.sim_time_params.time_range.day_of_year)
    BHI = GHI - DHI
    return BHI, DHI

def get_delta_azi(
        solar_azimuth: np.array,
        surface_azimuth: float,
        ):
    """
    get the difference between solar and surface azimuth angles.
    
    Args:
        solar_azimuth (float/array): Solar azimuth angle
        surface_azimuth (float): Surface azimuth angle
    
    Returns:
        float/array: Delta azimuth angle
    """
    delta = np.abs(solar_azimuth - surface_azimuth)
    delta = np.where(delta > 180, 360 - delta, delta)
    return np.where(delta <= 90, delta, 0)

def get_Norm_BHI(
        weather, 
        construction,
        ):
    """
    get beam radiation on tilted surface.
    
    Args:
        BHI (float/array): Beam Horizontal Irradiance
        delta_azimuth (float/array): Difference in azimuth angles
        solar_altitude (float/array): Solar altitude angle
        surface_tilt (float): Surface tilt angle
    
    Returns:
        float/array: Beam radiation on tilted surface
    """
    sol_alt = weather.sol_alt
    BHI = weather.BHI
    surface_tilt = construction.tilt
    delta_azimuth = get_delta_azi(weather.sol_azi, construction.azimuth)

    return np.where(sol_alt < 5, 0,
                   np.maximum(0, np.sin(d2r * sol_alt + d2r * surface_tilt) /
                            np.sin(d2r * sol_alt) * np.cos(d2r * delta_azimuth) * BHI))


## 4.4. Total Solar Radiation on Tilted Surface
def solar_to_unit_surface(
                        weather,
                        construction,
                        ):
    
    """
    get total solar radiation on a tilted surface.
    
    Args:
        BHI (array-like): Beam Horizontal Irradiance (W/m²)
        DHI (array-like): Diffuse Horizontal Irradiance (W/m²)
        station (str): Observatory name
        SimulationTimeParameters (object): Simulation time information
        surface_tilt (float): Surface tilt angle (degrees)
        surface_azimuth (float): Surface azimuth angle (degrees)
    
    Returns:
        array-like: Total solar radiation on tilted surface (W/m²)
    """

    BHI = weather.BHI
    DHI = weather.DHI
    
    surf_tilt = construction.tilt
    VF = (1 + np.cos(d2r * surf_tilt))/2

    DHI_surf = np.maximum(0, VF * DHI)
    BHI_surf = get_Norm_BHI(weather, construction)
    
    return np.maximum(0, BHI_surf + DHI_surf)