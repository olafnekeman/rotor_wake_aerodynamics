# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:33:38 2020

@author: Peter Blom
"""

'''
-------------------------------- Imports -----------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


'''
------------------------------- Supporting functions -----------------------------
'''

#function to get cartesian coordinates from polar coordinates
def get_cp_coords(xi,R_dist, plot=False):
    y_cp = R_dist*np.sin(xi)
    z_cp = R_dist*np.cos(xi)
    x_cp = np.zeros(len(R_dist))
    
    if plot:
        circle1 = plt.Circle((0,0),50,color='r')
        fig, ax = plt.subplots()
        #note, plot is from the front and positive y is to the left, thus y needs to be flipped in the plot
        ax.plot(-y_cp,z_cp,'o')
        ax.plot([0],[0],'bx')
        R = max(R_dist)
        plt.xlim([-R,R])
        plt.ylim([-R,R])
        ax.add_artist(circle1)
        
    return x_cp,y_cp,z_cp

#function to calculate the difference in cartesian coordinates along the chord line with twist beta
#the point on the chord line from the c/4 position is c, the twist is beta in radians and xi in radians
#is the angle of the blade in the y-z plane
def find_wake_point2(c,beta,xi):
    dx = np.sin(beta)*c
    dy = np.cos(xi)*np.cos(beta)*c
    dz = -np.sin(xi)*np.cos(beta)*c
    return dx,dy,dz



#Function to calculate the induction in coorp from a line starting at coorv1 and ending in coorv2
def calculate_induction(coorv1,coorv2,coorp,CORE = 0.00001):
    x1 = coorv1[0]
    y1 = coorv1[1]
    z1 = coorv1[2]
    x2 = coorv2[0]
    y2 = coorv2[1]
    z2 = coorv2[2]
    xp = coorp[0]
    yp = coorp[1]
    zp = coorp[2]
    
    
    R1 = np.sqrt((xp-x1)**2+(yp-y1)**2+(zp-z1)**2)
    R2 = np.sqrt((xp-x2)**2+(yp-y2)**2+(zp-z2)**2)
    R12x = (yp-y1)*(zp-z2)-(zp-z1)*(yp-y2)
    R12y = -(xp-x1)*(zp-z2)+(zp-z1)*(xp-x2)
    R12z = (xp-x1)*(yp-y2)-(yp-y1)*(xp-x2)    
    R12sqr = R12x**2+R12y**2+R12z**2
    R01 = (x2-x1)*(xp-x1)+(y2-y1)*(yp-y1)+(z2-z1)*(zp-z1)
    R02 = (x2-x1)*(xp-x2)+(y2-y1)*(yp-y2)+(z2-z1)*(zp-z2)
    
    if R1 < CORE:
        R1 = CORE
    if R2 < CORE:
        R2 = CORE
    if R12sqr < CORE**2:
        R12sqr = CORE**2
        
    K = 1/(4*np.pi*R12sqr)*(R01/R1-R02/R2)
    
    return np.array([K*R12x,K*R12y,K*R12z])


'''
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
    - gamma_bound_imper The spanwise distribution of the bound circulation from the analysis with impermeability boundary conditions
    - F_T               The spanwise distribution of the force in axial direction from this blade
    - F_az              The spanwise distribution of the force in azimuthal direction from this blade
    - M                 The spanwise distribution of the moment caused by each blade section
    
Important Functions:
    - __init__                      Function that intializes the blade and calculates the locations of the wake
    - calculate_wake_circulation    Function that calculates the circulation in the wake from the bound circulation
    - calculate_forces              Function that calculates the absolute forces in spanwise direction
    - calc_chord                    Function that interpolates the chord at a vector "mu" from the vectors "mu_vec" and "chord_vec"
    - calc_twist                    Function that interpolates the twist at a vector "mu" from the vectors "mu_vec" and "twist_vec"


'''

class blade:
    def __init__(self,R,mu_dist,mu_cent,mu_vec,chord_vec,twist_vec,xi,U_wake,N_wake_sec,dt,Omega,location = 0):
        N_blade_sec = len(mu_dist)-1
        self.R = R
        self.R_dist = mu_dist*R
        self.R_cent = mu_cent*R
        self.xi = xi
        self.location = location
        self.mu_vec = mu_vec
        self.chord_vec = chord_vec
        self.twist_vec = twist_vec
        self.c_dist = self.calc_chord(mu_dist)
        self.c_cent = self.calc_chord(mu_cent)
        
        self.twist_dist = self.calc_twist(mu_dist)
        self.twist_cent = self.calc_twist(mu_cent)
        
        self.alpha = np.zeros(N_blade_sec)
        self.inflowangle = np.zeros(N_blade_sec)
        self.fax = np.zeros(N_blade_sec)
        self.ftan = np.zeros(N_blade_sec)
        
        self.wake_induction_matrices_u = []
        self.wake_induction_matrices_v = []
        self.wake_induction_matrices_w = []
        
        self.bound_induction_matrices_u = []
        self.bound_induction_matrices_v = []
        self.bound_induction_matrices_w = []
        
        self.N_wake_sec = N_wake_sec
        self.N_blade_sec = N_blade_sec
        
        #initialize the bounded vorticity vector
        self.gamma_bound = np.zeros(N_blade_sec)
        
        #initialize the bounded vorticity vector for the impermeability b.c. solution
        self.gamma_bound_imper = np.zeros(N_blade_sec)
        
        #initialize the wake vorticity vector
        self.gamma_wake = np.zeros(N_wake_sec*(N_blade_sec+1))
        
        #get the locations of the control points for polar calculations
        self.x_cp,self.y_cp,self.z_cp = get_cp_coords(xi,mu_cent*R)
        
        #get the locations of the wake
        x_wake = np.zeros([N_blade_sec+1, N_wake_sec+1])
        y_wake = np.zeros([N_blade_sec+1, N_wake_sec+1])
        z_wake = np.zeros([N_blade_sec+1, N_wake_sec+1])
        
        #get the points along the blade that define the size of the blade sections in cartesian coordinates
        self.x_dist,self.y_dist,self.z_dist = get_cp_coords(xi,mu_dist*R)
        
        #run through the entire blade and calculate the wake locations
        for i in range(N_blade_sec+1):
            for j in range(N_wake_sec+1):
                if j == 0:
                    #the first point in the wake is always the one on the c/4 on the blade
                    x_wake[i,j] = self.x_dist[i]
                    y_wake[i,j] = self.y_dist[i]
                    z_wake[i,j] = self.z_dist[i]
                elif j == 1:
                    #the second point is always one which trails the blade one full chord behind the c/4
                    dx,dy,dz = find_wake_point2(self.c_dist[i],self.twist_dist[i],xi)
                    x_wake[i,j] = self.x_dist[i]+dx
                    y_wake[i,j] = self.y_dist[i]+dy
                    z_wake[i,j] = self.z_dist[i]+dz
                else:
                    #the other points are convected through the domain
                    t = (j-1)*dt
                    R_airfoil_end = np.sqrt(y_wake[i,1]**2+z_wake[i,1]**2)
                    if z_wake[i,1] > 0:
                        xi_airfoil_end = np.arctan(y_wake[i,1]/z_wake[i,1])   
                    elif z_wake[i,1] <0:
                        xi_airfoil_end = np.pi+np.arctan(y_wake[i,1]/z_wake[i,1])  

                    x_wake[i,j] = x_wake[i,1]+t*U_wake
                    y_wake[i,j] = R_airfoil_end*np.sin(xi_airfoil_end+Omega*t) 
                    z_wake[i,j] = R_airfoil_end*np.cos(xi_airfoil_end+Omega*t)
                    
        #calculate the normal vector to the blade in azimuthal direction
        self.n = [0,np.cos(xi),-np.sin(xi)]
        
        #get the locations of the control points for permeability boundary conditions
        self.x_cp_perm = np.zeros(self.N_blade_sec)
        self.y_cp_perm = np.zeros(self.N_blade_sec)
        self.z_cp_perm = np.zeros(self.N_blade_sec)
        
        #the control points are the ones that are c/2 behind the c/4 position, therefore the 
        #function find_wake_point2 can be used again
        for i in range(self.N_blade_sec):
            dx,dy,dz = find_wake_point2(self.c_cent[i]/2,self.twist_cent[i],xi)
            
            self.x_cp_perm[i] = self.x_cp[i]+dx
            self.y_cp_perm[i] = self.y_cp[i]+dy
            self.z_cp_perm[i] = self.z_cp[i]+dz

        #add the y-location of the rotor to the coordinates and save the wake coordinates
        self.x_wake = x_wake
        self.y_wake = y_wake+location
        self.z_wake = z_wake
        
        self.y_cp = self.y_cp + location
        self.y_dist = self.y_dist + location
        self.y_cp_perm = self.y_cp_perm + location
 
    def calculate_wake_circ(self):
        for i in range(self.N_blade_sec+1):
            indeces = np.arange((self.N_wake_sec)*i,(self.N_wake_sec)*i+self.N_wake_sec)
            if i == 0:
                self.gamma_wake[indeces] = -self.gamma_bound[i]*np.ones(self.N_wake_sec)
            elif i == self.N_blade_sec:
                self.gamma_wake[indeces] = self.gamma_bound[i-1]*np.ones(self.N_wake_sec)                
            else:
                self.gamma_wake[indeces] = (self.gamma_bound[i-1]-self.gamma_bound[i])*np.ones(self.N_wake_sec)
                
    def calculate_forces(self):
        self.F_T = np.zeros(self.N_blade_sec)
        self.F_az = np.zeros(self.N_blade_sec)
        self.M = np.zeros(self.N_blade_sec)
        
        for i in range(self.N_blade_sec):
            self.F_T[i] = (self.R_dist[i+1]-self.R_dist[i])*self.fax[i]
            self.F_az[i] = (self.R_dist[i+1]-self.R_dist[i])*self.ftan[i]
            self.M[i] = (self.R_dist[i+1]-self.R_dist[i])*self.ftan[i]*self.R_cent[i]
            
    def calc_chord(self,mu):
        return np.interp(mu,self.mu_vec,self.chord_vec)
    
    def calc_twist(self,mu):
        return np.interp(mu,self.mu_vec,self.twist_vec)*np.pi/180 


'''
------------------------------- Class lifting_line_model_singular  -----------------------------
This class models the performance of a single rotor placed in a uniform velocity field using 
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
    - solve_system_imper    Function to call to solve the system using impermeability boundary conditions
'''
class lifting_line_model_singular:
    def __init__(self, R, N_blades, mu_start, mu_end, mu_vec, chord_vec, twist_vec, U_0, TSR, N_blade_sec, N_wake_sec_rot, N_rot, a_w, airfoil, spacing = 'cosine'):
        #initialize the parameters
        self.R = R
        self.N_blades = N_blades
        self.mu_start = mu_start
        self.mu_end = mu_end
        self.U_0 = U_0
        self.TSR = TSR
        self.N_blade_sec = N_blade_sec
        self.N_wake_sec_rot = N_wake_sec_rot
        self.N_rot = N_rot
        self.a_w = a_w
        self.airfoil = airfoil
        self.mu_vec = mu_vec
        self.chord_vec = chord_vec
        self.twist_vec = twist_vec
        self.spacing = spacing
        
        #changeable parameters
        self.max_iter = 1000
        self.core_size = 0.25  #the size of the viscous core as a fracton of the length of a filament
        self.convergence_crit = 0.0001 #fraction of maximum change in bound vortex strength 
        self.weight_step = 0.25  #fraction of the updated gamma
        
        self.print_progress = True
        
    def commit_parameters(self):
        if self.print_progress:
            print('Initializing')
        
        data1=pd.read_csv(self.airfoil, header=0,
            names = ["alfa", "cl", "cd", "cm"],  sep='\s+')
        
        #flip the dataframe to make sure alpha will be increasing for the np.interp function in the propeller case
        if self.airfoil == 'ARA_polar.txt':
            data1 = data1.iloc[::-1]
            
        self.polar_alpha = data1['alfa'][:]
        self.polar_cl = data1['cl'][:]
        self.polar_cd = data1['cd'][:]

        if self.airfoil == 'ARA_polar.txt':
            #make both the cl and the alpha negative for the propeller case
            self.polar_alpha = -1*self.polar_alpha
            self.polar_cl = -1*self.polar_cl
        
        
        #calculate the rotational velocity
        self.Omega = self.TSR*self.U_0/self.R
        
        #calcalate the number of wake sections per blade
        self.N_wake_sec = self.N_rot*self.N_wake_sec_rot+1
        
        #calculate the time it takes for one full rotation 
        t_rot = 2*np.pi/self.Omega
        #calculate the time it takes between two vortex filaments in the wake
        dt = t_rot/self.N_wake_sec_rot
        
        #calculate the wake convection speed
        self.U_wake = self.U_0*(1-self.a_w)
        if self.print_progress:
            print('Setting up blade geometry')
            
        #create the blade division
        if self.spacing == 'cosine':
            beta_cosine = np.linspace(0,np.pi,self.N_blade_sec+1)
            mu_dist = self.mu_start+(self.mu_end-self.mu_start)/2*(1-np.cos(beta_cosine))
        
        elif self.spacing == 'uniform':
            mu_dist = np.linspace(self.mu_start,self.mu_end,self.N_blade_sec+1)
        else:
            raise ValueError('The spacing method is not recognized, please choose: "uniform" or "cosine"')
        
        #calculate the locations of the centroids
        mu_cp = np.zeros(self.N_blade_sec)
        for i in range(self.N_blade_sec):
            mu_cp[i] = (mu_dist[i]+mu_dist[i+1])/2
            
        #make the blade division            
        xi_blades = np.linspace(0,2*np.pi,self.N_blades, endpoint = False)
        self.blades = []
        for i in range(self.N_blades):
            if self.print_progress:
                print('Constructing blade number {}'.format(i))
                
            self.blades.append(blade(self.R,mu_dist,mu_cp,self.mu_vec,self.chord_vec,self.twist_vec,xi_blades[i],self.U_wake,self.N_wake_sec,dt, self.Omega))
        
        if self.print_progress:
            print('Setting up wake induction matrices')
        
        #in case of only one rotor, only the first blade needs to be considered
        for i in [0]:#range(len(self.blades)):#loop through all blades to construct matrices for all control points
            for j in range(len(self.blades)):#loop through all blades to construct the wake and bounded matrices for the current blade
                
                if self.print_progress:
                    print('Evaluating the effect of the vorticity from blade {} on blade {}'.format(j,i))
                
                wake_matrix_u = np.zeros([self.N_blade_sec,self.N_wake_sec*(self.N_blade_sec+1)])
                wake_matrix_v = np.zeros([self.N_blade_sec,self.N_wake_sec*(self.N_blade_sec+1)])
                wake_matrix_w = np.zeros([self.N_blade_sec,self.N_wake_sec*(self.N_blade_sec+1)])
                
                bound_matrix_u = np.zeros([self.N_blade_sec,self.N_blade_sec])
                bound_matrix_v = np.zeros([self.N_blade_sec,self.N_blade_sec])
                bound_matrix_w = np.zeros([self.N_blade_sec,self.N_blade_sec])
                
                #loop through all the blade sections in this blade to evaluate the effect of 
                #a unit vorticity from blade j on each control point
                for k in range(self.N_blade_sec):
                    column_number = 0
                    coorp = [self.blades[i].x_cp[k],self.blades[i].y_cp[k],self.blades[i].z_cp[k]]
                    #run through all wake sections from the blade j 
                    for i1 in range(self.N_blade_sec+1):
                        for j1 in range(self.N_wake_sec):
                            coor1 = [self.blades[j].x_wake[i1,j1],self.blades[j].y_wake[i1,j1],self.blades[j].z_wake[i1,j1]]
                            coor2 = [self.blades[j].x_wake[i1,j1+1],self.blades[j].y_wake[i1,j1+1],self.blades[j].z_wake[i1,j1+1]]
                            #determine the viscous core size
                            CORE =  self.core_size*np.sqrt((coor1[0]-coor2[0])**2+(coor1[1]-coor2[1])**2+(coor1[2]-coor2[2])**2)
                            wake_matrix_u[k,column_number],wake_matrix_v[k,column_number],wake_matrix_w[k,column_number] = calculate_induction(coor1,coor2,coorp,CORE=CORE)
                            column_number += 1
                    #do the same for all the bound vortices from blade j    
                    for k1 in range(self.N_blade_sec):
                        coor1 = [self.blades[j].x_dist[k1],self.blades[j].y_dist[k1],self.blades[j].z_dist[k1]]
                        coor2 = [self.blades[j].x_dist[k1+1],self.blades[j].y_dist[k1+1],self.blades[j].z_dist[k1+1]]
                        CORE = self.core_size*np.sqrt((coor1[0]-coor2[0])**2+(coor1[1]-coor2[1])**2+(coor1[2]-coor2[2])**2)
                        bound_matrix_u[k,k1],bound_matrix_v[k,k1],bound_matrix_w[k,k1] = calculate_induction(coor1,coor2,coorp, CORE=CORE)
                
                #append these induction matrices to the blade classes to save for later use
                self.blades[i].wake_induction_matrices_u.append(wake_matrix_u)
                self.blades[i].wake_induction_matrices_v.append(wake_matrix_v)
                self.blades[i].wake_induction_matrices_w.append(wake_matrix_w)
                
                self.blades[i].bound_induction_matrices_u.append(bound_matrix_u)
                self.blades[i].bound_induction_matrices_v.append(bound_matrix_v)
                self.blades[i].bound_induction_matrices_w.append(bound_matrix_w)
    
    def solve_system(self):
        if self.print_progress:
            print('Solving problem using lift and drag polar data')
        
        #start the iteration loop
        for iteration in range(self.max_iter):
            #set up the matrices which track the relative differences in bounded circulations
            error = np.zeros([len(self.blades),self.N_blade_sec])
            
            #calculate the wake circulation for each blade
            for i in range(len(self.blades)):
                self.blades[i].calculate_wake_circ()
            
            #evaluate the bound vorticity and induction in the first blade
            for i in [0]:#range(len(self.blades)):
                #matrices with the u,v and w induction in each control point
                self.blades[i].cp_induction_u = np.zeros(self.N_blade_sec)
                self.blades[i].cp_induction_v = np.zeros(self.N_blade_sec)
                self.blades[i].cp_induction_w = np.zeros(self.N_blade_sec)
                
                #run through each blade and find the induction in each control point
                for j in range(len(self.blades)):
                    self.blades[i].cp_induction_u = self.blades[i].cp_induction_u + np.matmul(self.blades[i].wake_induction_matrices_u[j],self.blades[j].gamma_wake)
                    self.blades[i].cp_induction_u = self.blades[i].cp_induction_u + np.matmul(self.blades[i].bound_induction_matrices_u[j],self.blades[j].gamma_bound)

                    self.blades[i].cp_induction_v = self.blades[i].cp_induction_v + np.matmul(self.blades[i].wake_induction_matrices_v[j],self.blades[j].gamma_wake)
                    self.blades[i].cp_induction_v = self.blades[i].cp_induction_v + np.matmul(self.blades[i].bound_induction_matrices_v[j],self.blades[j].gamma_bound)

                    self.blades[i].cp_induction_w = self.blades[i].cp_induction_w + np.matmul(self.blades[i].wake_induction_matrices_w[j],self.blades[j].gamma_wake)
                    self.blades[i].cp_induction_w = self.blades[i].cp_induction_w + np.matmul(self.blades[i].bound_induction_matrices_w[j],self.blades[j].gamma_bound)
                
                #determine the forces for each blade section
                for k in range(self.N_blade_sec):
                    U_ax = self.blades[i].cp_induction_u[k]+self.U_0
                    tan_induction = np.dot([U_ax, self.blades[i].cp_induction_v[k], self.blades[i].cp_induction_w[k]],self.blades[i].n)
                    U_tan = self.Omega*self.blades[i].R_cent[k]+tan_induction
                    
                    V_p = np.sqrt(U_ax**2+U_tan**2)

                    inflowangle = np.arctan2(U_ax,U_tan)
                    self.blades[i].inflowangle[k] = inflowangle
                    alpha = inflowangle*180/np.pi - self.blades[i].twist_cent[k]*180/np.pi
                    self.blades[i].alpha[k] = alpha

                    cl = np.interp(alpha, self.polar_alpha, self.polar_cl)
                    cd = np.interp(alpha, self.polar_alpha, self.polar_cd)
                    lift = 0.5*V_p**2*cl*self.blades[i].c_cent[k]
                    drag = 0.5*V_p**2*cd*self.blades[i].c_cent[k]
                    fnorm = lift*np.cos(inflowangle)+drag*np.sin(inflowangle)
                    ftan = lift*np.sin(inflowangle)-drag*np.cos(inflowangle)
                    self.blades[i].fax[k] = fnorm
                    self.blades[i].ftan[k] = ftan
                    
                    #determine the new value for circulation
                    gamma_new = 0.5*V_p*cl*self.blades[i].c_cent[k]
                    
                    #evaluate the difference in circulation with the previous step
                    error[i,k] = abs(self.blades[i].gamma_bound[k]-gamma_new)/max([abs(self.blades[i].gamma_bound[k]),0.001])
                    #update the bound circulation in each blade with the newly found value
                    for index in range(self.N_blades):
                        self.blades[index].gamma_bound[k] = (1-self.weight_step)*(self.blades[i].gamma_bound[k]) + self.weight_step*gamma_new
     
            #if the convergence criterium is met in every point, the model has converged
            if max(error.flatten()) < self.convergence_crit:
                print('Converged within iteration limit!')
                break
        #raise an error if the model has not converged within the maximum iterations
        if iteration == self.max_iter-1:
            print('WARNING: Model did not converge within the convergence criteria given.')
        
        #calculate the total force distribution
        self.blades[i].calculate_forces()
        total_T = self.N_blades*np.sum(self.blades[i].F_T)
        total_M = self.N_blades*np.sum(self.blades[i].M)
            
        #calculate and print the results
        self.CT_polar = total_T/(0.5*self.U_0**2*np.pi*self.R**2)
        self.CP_polar = total_M*self.Omega/(0.5*self.U_0**3*np.pi*self.R**2)
        print('CT = {}'.format(self.CT_polar))
        print('CP = {}'.format(self.CP_polar))
        
    def solve_system_imper(self):
        if self.print_progress:
            print('Solving system using impermeability boundary conditions')
            print('Setting up the system of equations modelling the effect of all the values of vorticity on the control points')       
        
        A = np.zeros([self.N_blade_sec*len(self.blades),self.N_blade_sec*len(self.blades)])
        rhs = np.zeros(self.N_blade_sec*self.N_blades)
        
        row_number = 0
        #determine the angle of attack with zero lift of the airfoil
        alpha_0 = np.interp(0,self.polar_cl,self.polar_alpha) 
        
        #run through the control points in each blade each corresponding to a row in matrix A
        for i in range(len(self.blades)):
            for j in range(self.N_blade_sec):
                if self.print_progress:
                    print('Evaluating the induction in the {}th controlpoint of blade {}'.format(j,i))
                
                #find the coordinates of the current control point
                coorp = [self.blades[i].x_cp_perm[j],self.blades[i].y_cp_perm[j],self.blades[i].z_cp_perm[j]]
                
                #determine the local twist of the section and correct it with the zero-lift angle of attack
                #such that the airfoil can be assumed a flat plate from here on out.
                twist = self.blades[i].twist_cent[j] + alpha_0*np.pi/180
                radius = self.blades[i].R_cent[j]
                
                #run through each bound vortex on each blade and find the induction caused by this bound vortex
                #including the induction by the wake terms on the control point
                column_number = 0
                for k in range(len(self.blades)): #each blade
                    for i1 in range(self.N_blade_sec): #each bound vortex in each blade
                        total_induction = np.zeros(3)
                        #run through the bottom line of the horseshoe vortex created by the current bound vortex and add induction
                        #to the current control point 
                        for j1 in range(self.N_wake_sec):
                            coor1 = [self.blades[k].x_wake[i1,j1],self.blades[k].y_wake[i1,j1], self.blades[k].z_wake[i1,j1]]
                            coor2 = [self.blades[k].x_wake[i1,j1+1],self.blades[k].y_wake[i1,j1+1], self.blades[k].z_wake[i1,j1+1]]
                        
                            CORE =  self.core_size*np.sqrt((coor1[0]-coor2[0])**2+(coor1[1]-coor2[1])**2+(coor1[2]-coor2[2])**2)
                            
                            temp_induction_u,temp_induction_v,temp_induction_w = calculate_induction(coor1,coor2,coorp,CORE=CORE)
                            
                            total_induction = total_induction - np.array([temp_induction_u,temp_induction_v,temp_induction_w])
                        #run through the top line of the horseshoe vortex created by the current bound vortex and add induction
                        #to the current control point
                        for j1 in range(self.N_wake_sec):
                            coor1 = [self.blades[k].x_wake[i1+1,j1],self.blades[k].y_wake[i1+1,j1], self.blades[k].z_wake[i1+1,j1]]
                            coor2 = [self.blades[k].x_wake[i1+1,j1+1],self.blades[k].y_wake[i1+1,j1+1], self.blades[k].z_wake[i1+1,j1+1]]
                        
                            CORE =  self.core_size*np.sqrt((coor1[0]-coor2[0])**2+(coor1[1]-coor2[1])**2+(coor1[2]-coor2[2])**2)
                            
                            temp_induction_u,temp_induction_v,temp_induction_w = calculate_induction(coor1,coor2,coorp,CORE=CORE)
                            
                            total_induction = total_induction + np.array([temp_induction_u,temp_induction_v,temp_induction_w])
                        
                        #add the contribution of the bound vortex itself
                        coor1 = [self.blades[k].x_dist[i1],self.blades[k].y_dist[i1],self.blades[k].z_dist[i1]]
                        coor2 = [self.blades[k].x_dist[i1+1],self.blades[k].y_dist[i1+1],self.blades[k].z_dist[i1+1]]
                        
                        CORE =  self.core_size*np.sqrt((coor1[0]-coor2[0])**2+(coor1[1]-coor2[1])**2+(coor1[2]-coor2[2])**2)
                            
                        temp_induction_u,temp_induction_v,temp_induction_w = calculate_induction(coor1,coor2,coorp,CORE=CORE)

                        total_induction = total_induction + np.array([temp_induction_u,temp_induction_v,temp_induction_w])
                        
                        A[row_number,column_number] = total_induction[0]*np.cos(twist)-np.dot(total_induction,self.blades[i].n)*np.sin(twist)
                        #A[row_number,column_number] = total_induction[0]+np.dot(total_induction,self.blades[i].n)
                        #print((total_induction[0]*np.cos(twist)-np.dot(total_induction,self.blades[i].n)*np.sin(twist))/(total_induction[0]+np.dot(total_induction,self.blades[i].n)))
                        column_number += 1
                
                rhs[row_number] = -self.U_0*np.cos(twist)+self.Omega*np.sin(twist)*radius

                row_number += 1
        
        if self.print_progress:
            print('Solving system of equations for the bounded vorticity')
            
        A_inv = np.linalg.inv(A)       
        self.gamma_res_bound = np.matmul(A_inv,rhs)
        
        for i in range(len(self.blades)):
            for k in range(self.N_blade_sec):
                self.blades[i].gamma_bound_imper[k] = self.gamma_res_bound[self.N_blade_sec*i+k]
                



