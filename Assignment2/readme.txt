This document contains the inputs, important attributes and important functions of the two 
classes used in the lifting line model. Note that a '+' denotes that the input, attribute
or function is only present in case of the 'lifting_line_model'. If a '*' is present, the 
input, attribute or function is only present for the 'lifting_line_model_singular'.

The results for overall CT and CP can be found in the lifting_line_model classes, whereas
the distribution of angle of attack, inflowangle, axial & azimuthal forces and circulation
can be found in the instances of the class 'blade' saved in the list 
'lifting_line_model.blades'. 

For more details, read the list of important attributes of the classes below:

---------------- Class lifting_line_model and lifting_line_model_singular  -----------------------------
This class models the performance of multiple or a single rotor placed in a uniform velocity field using 
a lifting line model.

Inputs:
    - R                 Absolute Radius of the blade
    - N_blades          The number of blades of the rotor
    - mu_start          The normalized radial position on which the blade starts
    - mu_end            The normalized radial position on which the blade ends
    - mu_vec            Normalized radial locations in which the chord and twist distribution along the blade is known 
    - chord_vec         Absolute value of the chord in the locations specified by mu_vec               
    - twist_vec         Absolute value of the twist in degree in the locations specified by mu_vec 
    - U_0               The value of the freestream velocity
    - TSR               The tip speed ratio of the rotor
    - N_blade_sec       The number of blade sections
    - N_wake_sec_rot    The number of wake sections in the axial direction per rotation of the blade
    - N_rot             The number of rotations the blade has made determining the length of the wake (NOTE needs to be an integer)
    - a_w               The assumed wake convection induction factor (i.e. U_wake = U_0*(1-a_w))
    - airfoil           The name of the file containing the airfoil polar
    - spacing           The spacing method to use. Choice from "cosine" or "uniform"
    + N_rotors          The number of rotors in the domain (currently capped at 2)
    + phase_diff        If N_rotors =2, phase_diff is the phase difference between the two rotors.
    + location          If N_rotors =2, location is the list with the y_locations of the rotors, has to be a list with length 2 in that case

Important Attributes:
    - blades            The list containing all the instances of the class 'blade', to be used to get the results 
    - CT_polar          The thrust coefficient of the rotor from the polar analysis
    - CP_polar          The power coefficient of the rotor from the polar analysis
    - max_iter          The maximum iterations the model should perform 
    - core_size         The size of the viscous core as a fraction of the length of a filament
    - convergence_crit  Fraction of the maximum change in bound vortex strength for the convergence criterium
    - weight_step       The fraction to which the new bound vorticity of each blade is updated with the calculated, needs to be between 0 and 1 (i.e. gamma_bound = (1-weight_step)*gamma_bound+weight_step*gamma_bound)
    - print_progress    The setting turning on and off what the program is doing, i.e. True or False

Important Functions:
    - commit_parameters     Function to call before you want to run a simulation, this function sets up the geometry of the wake and blades
    - solve_system          Function to call to solve the system using polar data given by airfoil
    * solve_system_imper    Function to call to solve the system using impermeability boundary conditions

------------------------------- Class blade housing the blade geometries -----------------------------
Inputs:
     - R                Absolute Radius of the blade
     - mu_dist          Normalized radial locations of the points along the blade
     - mu_cent          Normalized radial locations of the centroids of the blade sections
     - mu_vec           Normalized radial locations in which the chord and twist distribution along the blade is known 
     - chord_vec        Absolute value of the chord in the locations specified by mu_vec               
     - twist_vec        Absolute value of the twist in degree in the locations specified by mu_vec       
     - xi               Angle of the blade in the y-z plane 
     - U_wake           Velocity of the wake
     - N_wake_sec       Number of wake sections in axial direction (the total # wake sections is N_wake_sec*N_blade_sec)
     - dt               The time difference between two wake sections
     - Omega            The rotational speed of the blade in rad/s
     - location         The absolute y-location of the rotor
     
Important Attributes:
    - inflowangle       The spanwise distribution of the inflowangle in rad
    - alpha             The spanwise distribution of the angle of attack in deg
    - gamma_bound       The spanwise distribution of the bound circulation using polar data
    - R_cent            The absolute radial positions of the centroids of the blade sections
    * gamma_bound_imper The spanwise distribution of the bound circulation from the analysis with impermeability boundary conditions
    - F_T               The spanwise distribution of the force in axial direction from this blade
    - F_az              The spanwise distribution of the force in azimuthal direction from this blade
    - M                 The spanwise distribution of the moment caused by each blade section
    
Important Functions:
    - __init__                      Function that intializes the blade and calculates the locations of the wake
    - calculate_wake_circulation    Function that calculates the circulation in the wake from the bound circulation
    - calculate_forces              Function that calculates the absolute forces in spanwise direction
    - calc_chord                    Function that interpolates the chord at a vector "mu" from the vectors "mu_vec" and "chord_vec"
    - calc_twist                    Function that interpolates the twist at a vector "mu" from the vectors "mu_vec" and "twist_vec"

