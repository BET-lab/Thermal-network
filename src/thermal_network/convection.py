# exeterior convection coefficient 
def simple_combined_convection(roughness, Vz):
    roughness_values = {
    'very rough': (11.58, 5.894, 0.0),
    'rough': (12.49, 4.065, 0.028),
    'medium rough': (10.79, 4.192, 0.0),
    'medium smooth': (8.23, 4.0, -0.057),
    'smooth': (10.22, 3.1, 0.0),
    'very smooth': (8.23, 3.33, -0.036)
    }
    if roughness in roughness_values:
        D, E, F = roughness_values[roughness]
        h_co = D + E*Vz + F*Vz**2
    else:
        raise ValueError("Invalid roughness value")
    return h_co

