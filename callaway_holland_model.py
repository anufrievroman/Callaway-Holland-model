# This python code is written by Dr. Roman Anufriev for Nomura lab, IIS, University of Tokyo in 2018-2019
# The code implements Callaway-Holland model which calculates the thermal conductivity of nanostructures.
# Contact me by anufriev.roman@protonmail.com if you have any questions.

from numpy import pi, exp, zeros, savetxt, sqrt, sinh
import matplotlib.pyplot as plt
from scipy.constants import k, hbar

# PARAMETERS
N = 1000            # Number of points
ROUGHNESS = 2.0e-9  # [nm]
MATERIAL = 'Si'


def bulk_phonon_dispersion(N):
    '''Return phonon dispersion calculated for N wavevectors over the G-X direction'''
    dispersion = zeros((N,4))						    # Ref. APL 95 161901 (2009)
    dispersion[:,0] = [k*12e9/(N-1) for k in range(N)]			    # Column of wavevectors
    if MATERIAL == 'Si':
        dispersion[:,1] = [abs(1369.42*k-2.405e-8*(k**2)-9.70e-19*(k**3)) for k in dispersion[:,0]] # LA branch
        dispersion[:,2] = [abs(1081.74*k-7.711e-8*(k**2)+5.674e-19*(k**3)+7.967e-29*(k**4)) for k in dispersion[:,0]]  # TA branch
        dispersion[:,3] = dispersion[:,2]					    # TA branch 
    return dispersion    


def phonon_properties_assignment_2(j, branch, N, dispersion):
    '''Assign phonon frequency (f) according to provided wavevector and branch, and calculate group velocity from bulk disperion'''
    K = (dispersion[j+1,0]+dispersion[j,0])/2.0				    # Wavevector (we take average in the interval)
    dK = (dispersion[j+1,0]-dispersion[j,0])				    # Delta wavevector
    w = 2.0*pi*abs((dispersion[j+1,branch+1]+dispersion[j,branch+1])/2.0)   # Angular freequency  (we take average in the interval)
    dw = 2.0*pi*abs(dispersion[j+1,branch+1]-dispersion[j,branch+1])	    # Delta angular freequency        
    speed = dw/dK                                                           # Group velocity
    frequency = w/(2*pi)   
    polarization = 'LA' if branch == 0 else 'TA'                            # Polarization according to the branch
    return frequency, polarization, speed, w, K, dK


def relaxation_times(K, w, L_parameter, speed, T, p, polarization):
    '''This function calculates relaxation times and summs them according to the Mathiesen's rule'''
    A = 2.95e-45                                                            # Parameters from bulk fitting                                                            
    B = 0.95e-19                                                              
    #E=3.3e-3								    # Crystal boundary 
    deb_temp = 152.0                                                          # Debye temperature
    time_i = A*(w**4.0)                                                     # 1/Impurities scattering
    time_u = B*(w**2.0)*T*exp(-deb_temp/T)                                  # 1/Umklapp scattering
    #if polarization=='TA':
    #    time_u=((3.28e-19)*(w**2)*T*exp(-140/T))/4			    # ref. JAP 110 034308 (2011)
    #if polarization=='LA':
    #    time_u=((3.28e-19)*(w**2)*T*exp(-140/T))  
    time_h = (speed/L_parameter)*(1.0-p)/(1.0+p)                            # scattering in the necks
    time_relaxation = 1.0/(time_i+time_u+time_h)                            # Total relaxation time
    return time_relaxation


def main():
    '''This is the main function, which calculates the thermal conductivity'''
    # material_density = 2330 #[kg/m^3]
    dispersion = bulk_phonon_dispersion(N+1)
    temperatures = range(200, 405, 5)
    #Lc=[70e-9, 100e-9, 150e-9, 200e-9, 250e-9, 300e-9, 350e-9, 400e-9, 500e-9]
    thermal_conductivities = zeros((len(temperatures),2)) 
    specific_heat_capacities = zeros((len(temperatures),2)) 
    #L_parameter=1.12*sqrt((145e-9)*(150e-9))       # This is the limiting dimension, i.e. neck of PnC or NW width etc.
    L_parameter=sqrt(2)*200e-9
    print("Please wait...")

    for index, T in enumerate(temperatures):
    # for index, L_parameter in enumerate(Lc):
        thermal_conductivity = 0.0
        total_heat_capacity = 0.0
        for branch in range(3):                                             # For each phonon branch
            for j in range(N):    
                frequency,polarization,speed,w,K,dK = phonon_properties_assignment_2(j,branch,N,dispersion)
                p = exp(-4.0*(K**2.0)*ROUGHNESS**2.0)                         # Specularity. Ref. APL 106, 133108 (2015)
                time_relaxation = relaxation_times(K,w,L_parameter,speed,T,p,polarization)                  
                heat_capacity = k*((hbar*w/(k*T))**2.0)*exp(hbar*w/(k*T))/((exp(hbar*w/(k*T))-1.0)**2.0) # Ref. PRB 88 155318 (2013)
                thermal_conductivity += (1.0/(6.0*(pi**2.0)))*heat_capacity*(speed**2.0)*time_relaxation*K**2.0*dK          
                total_heat_capacity += (1/(2*pi**2))*heat_capacity*K**2.0*dK    #[J/K] Total heat capacity integrated over all branches and wavevectors
                # mfp[j,0]=(w/(2*pi))*1e-12
                # mfp[j,branch+1]=speed*time_relaxation 

        thermal_conductivities[index,0] = T
        thermal_conductivities[index,1] = thermal_conductivity
        # specific_heat_capacities[index,0] = T
        # specific_heat_capacities[index,1] = total_heat_capacity/material_density # [J/kg/K]

    plt.plot(thermal_conductivities[:,0],thermal_conductivities[:,1])
    plt.ylabel('Thermal conductivity (W/mK)')
    plt.xlabel('Temperature (K)')
    plt.show()
    savetxt('thermal_conductivity.txt', thermal_conductivities, delimiter="	")

    #plt.loglog(mfp[:,0],mfp[:,1],mfp[:,0],mfp[:,2],mfp[:,0],mfp[:,3])
    #plt.ylabel('Mean Free Path (m)', fontsize=12)
    #plt.xlabel('Frequency (THz)', fontsize=12)
    #plt.show()
     
    #plt.plot(specific_heat_capacities[:,0], specific_heat_capacities[:,1])
    #plt.ylabel('Heat capacity (J/kg/K)', fontsize=12)
    #plt.xlabel('Temperature (K)', fontsize=12)
    #plt.show()

    #plt.plot(dispersion[:,0],dispersion[:,1],dispersion[:,0],dispersion[:,2],dispersion[:,0],dispersion[:,3])
    #plt.ylabel('Frequency (Hz)', fontsize=12)
    #plt.xlabel('Wavevector (1/m)', fontsize=12)
    #plt.show()


if __name__ == "__main__":
    main()
