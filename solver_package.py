import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.constants import c, pi, h, hbar

def solve(mypulse,mysystem):
    '''Solve the density matrix rate equation of a system interacting with a pulse.

    Parameters:
    -----------
    mypulse : pulse object or npulse object

    mysystem : system object

    Returns:
    --------
    rho : density matrix
    '''
    #temporal resolution is halved due to Runge Kutta 4:
    steps = mypulse.time.shape[0]
    steps_new = int((steps-1)/2)+1
    
    #generate 3D numpy array for values of rho and read in values for rho_0:
    rho_0 = np.zeros((mysystem.num,mysystem.num))
    for i in range(mysystem.num):
        rho_0[i,i] = mysystem.rho_init[i]
    rho = np.empty((mysystem.num,mysystem.num,steps_new), dtype=complex)
    rho[:,:,0] = rho_0    
    
    #stepsize is constant and does not have to be recalculated for each RK4 step:
    stepsize = mypulse.time[2]-mypulse.time[0]
    
    #generate empty numpy array:
    #this empty numpy array needs to be created at this point, because inside jit compiled functions np.empty() does not work
    drho_dt_empty = np.empty((mysystem.num,mysystem.num), dtype=complex)

    #RK4:
    if all(x<=y for x, y in zip(mysystem.energy, mysystem.energy[1:])):
        pass
    else:
        print('Solver terminated.\nSolver permits only systems with non decreasing energy list.')
        return
    ###if RWA is True (= pulserwa class and npulserwa class), the corresponding RK4rwa solver is used
    if mypulse._rwa:
        ###for npulserwa class: check if only one constant central wavelength for all spectra is used
        try:
            not_equal = False
            for i in mypulse.f_0:
                if mypulse.f_0[0] != i:
                    not_equal = True
                    break
                else:
                    pass
            if not_equal:
                print('Solver terminated.\nSolver in RWA mode permits only one constant central wavelength for all spectra.\nHowever, at least two different central wavelengths were detected.')
                return
            else:
                f_0 = mypulse.f_0[0]
        ###for pulserwa class: there is only one central wavelength
        except:
            f_0 = mypulse.f_0
        ###run solver:
        rho = RK4rwa(steps, stepsize, mysystem.num, mysystem.vibronic_trans, mysystem.vibrational_trans, mypulse.E_time, mysystem.gamma, mysystem.Gammabar, mysystem.mu, mysystem.f, f_0, drho_dt_empty, rho)
        print('Solver finished. RWA was used.')
    
    ###else RWA is False (= pulse class and npulse class), the corresponding RK4 solver is used
    else:
        ###run solver:
        rho = RK4(steps, stepsize, mysystem.num, mypulse.E_time, mysystem.gamma, mysystem.Gammabar, mysystem.mu, mysystem.f, drho_dt_empty, rho)
        print('Solver finished.')
    return rho

@jit(nopython=True, nogil=True)
def RK4(steps,stepsize,num,E_time,gamma,Gammabar,mu,f,drho_dt_empty,rho):
    for i in range(0,steps-2,2):
        i_new = int(i/2)
        k_1 = stepsize*rate_eq(num, rho[:,:,i_new], E_time[i], gamma, Gammabar, mu, f, drho_dt_empty,rho[:,:,0])
        k_2 = stepsize*rate_eq(num, rho[:,:,i_new]+0.5*k_1, E_time[i+1], gamma, Gammabar, mu, f, drho_dt_empty,rho[:,:,0])
        k_3 = stepsize*rate_eq(num, rho[:,:,i_new]+0.5*k_2, E_time[i+1], gamma, Gammabar, mu, f, drho_dt_empty,rho[:,:,0])
        k_4 = stepsize*rate_eq(num, rho[:,:,i_new]+k_3, E_time[i+2], gamma, Gammabar, mu, f, drho_dt_empty,rho[:,:,0])
        rho[:,:,i_new+1] = rho[:,:,i_new]+(k_1 + 2*k_2 + 2*k_3 + k_4)/6
    return rho

@jit(nopython=True, nogil=True)
def rate_eq(num,rho,E,gamma,Gammabar,mu,f,drho_dt,rho_0):
    for i in range(num):
        for j in range(num):
            
            #drho_dt[0,0] is calculated last and is skipped at this point:
            if (i,j) == (0,0):
                pass
            #calculate diagonal elements (drho_dt[i,j] for i=j):
           
            
            elif i == j:
                sum1 = -gamma[i,j]*rho[i,j]
                for k in range(i+1,num):
                    sum1 += gamma[i,k]*rho[k,k]
                sum2 = 0+0j
                for k in range(num):
                    if k == i or mu[k,i] == 0:
                        pass
                    elif rho_0[i,i]==rho[i,i]: #Keeps boltzmann with no E_field
                        sum1 = 0+0j
                        sum2 += mu[k,i]*(rho[k,i]-rho[i,k])
                    else:
                        sum2 += mu[k,i]*(rho[k,i]-rho[i,k])
                drho_dt[i,j] = sum1+1j*E/hbar*sum2
            #calculate non diagonal elements (drho_dt[i,j] for i<j):
            elif i < j:
                sum1 = -Gammabar[i,j]*rho[i,j] + 1j*f[j,i]*rho[i,j]
                sum2 = 0+0j
                if mu[i,j] == 0:
                    pass
                else:
                    sum2 += mu[i,j]*(rho[j,j]-rho[i,i])
                for k in range(num):
                    if k == j or k == i or mu[k,i] == 0:
                        pass
                    else:
                        sum2 += mu[k,i]*rho[k,j]
                for k in range(num):
                    if k == i or k == j or mu[k,j] == 0:
                        pass
                    else:
                        sum2 += -mu[k,j]*rho[i,k]
                        
                        
                drho_dt[i,j] = sum1+1j*E/hbar*sum2
            #calculate non diagonal elements (drho_dt[i,j] for i>j):
            else:
                drho_dt[i,j] = np.conj(drho_dt[j,i])
    #calculate drho_dt[0,0]:
    sum1 = 0+0j
    for i in range(1,num):
        sum1 += drho_dt[i,i]
    drho_dt[0,0] = -sum1
    return drho_dt


def plot(mypulse,mysystem,rho):
    '''Plot results.

    Parameters:
    -----------
    mypulse : pulse object or npulse object

    mysystem : system object

    rho : 3D numpy array
        Density matrix obtained by function solve(mypulse,mysystem)
    '''
    #rescale values since temporal resolution was halved due to Runge Kutta 4
    time_new = mypulse.time[::2]
    try:
        phase_time_new = mypulse.phase_time[::2]
    except:
        pass
    E_time_new = mypulse.E_time[::2]

    #calculate ground and excited state population:
    ground_state = np.zeros(time_new.shape[0])
    for i in range(mysystem.size[0]):
        ground_state += np.real(rho[i,i,:])
    
    excited_state = np.zeros(time_new.shape[0])
    for i in range(mysystem.size[0],mysystem.num):
        excited_state += np.real(rho[i,i,:])

    #calculate absorption:
    absorption = np.zeros(time_new.shape[0])
    for i in range(mysystem.num):
            for j in range(mysystem.num):
                if mysystem.vibronic_trans[i,j] == 1:
                    absorption += -np.imag(rho[i,j,:])
                else:
                    pass


    #plot:
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(6,12), sharex='col')
    ax[0].set_title('Pulse')
    ax[0].set_ylabel('E (V/m)')
    ax[0].plot(time_new, np.real(E_time_new), label='real')
    ax[0].plot(time_new, np.imag(E_time_new), label='imag')
    ax[0].legend(loc='best')

    ax[1].set_title('Phase')
    ax[1].set_ylabel('phase (rad s-1)')
    try:
        ax[1].plot(time_new, phase_time_new)
    except:
        pass

    ax[2].set_title('population of ground and excited state')
    ax[2].set_xlabel('time (s)')
    ax[2].set_ylabel('population')
    ax[2].plot(time_new, ground_state, label='ground state', color='tab:blue')
    ax3 = ax[2].twinx()
    ax3.plot(time_new, excited_state, label='excited state', color='tab:red')
    
    ax[3].set_title('Absorption')
    ax[3].set_xlabel('time (s)')
    ax[3].set_ylabel('absorption (a.u.)')
    ax[3].plot(time_new, absorption, label='real')
 

    fig.subplots_adjust(hspace=0.3)
    plt.show()


    fig, axb = plt.subplots(figsize=(8,6))
    axb.set_xlabel('time (s)')
    if isinstance(E_time_new[0], complex):
        axb.plot(time_new, np.abs(E_time_new)/np.max(np.abs(E_time_new)), label='Pulse norm.', color='r', alpha=0.6)
    elif isinstance(E_time_new[0], float):
        axb.plot(time_new, E_time_new/np.max(E_time_new), label='Pulse norm.', color='r', alpha=0.6)
    axb.plot(time_new, excited_state/np.max(excited_state), label='Excited state pop. norm.', color='k')
    axb.plot(time_new, absorption/np.max(np.abs(absorption)), label='Absorption norm.', color='b', alpha=0.6)
    axb.legend(loc='best')

    plt.show()
    return


def save(mypulse,rho,filename):
    '''Save pulse (pulse.E_time or npulse.E_time) and density matrix (rho) to text file

    Parameters:
    -----------
    mypulse : pulse object or npulse object

    rho : 3D numpy array
        Density matrix obtained by function solve(mypulse,mysystem)

    filename : string
        Name of the text file
    '''
    steps = rho.shape[2]
    num = rho.shape[0]

    time_new = np.reshape(mypulse.time[::2], (1,steps))
    E_time_new = np.reshape(mypulse.E_time[::2], (1,steps))

    rho_new = np.reshape(rho,(num**2,steps))

    save = np.concatenate((time_new,E_time_new,rho_new), axis=0)

    header_str = 'time (s), E (V m-1)'
    for i in range(num):
        for j in range(num):
            header_str += ', rho['+str(i)+','+str(j)+']'
    
    np.savetxt(filename+'.txt', save.T, delimiter=',', header=header_str, comments='')

    print('File saved.')
    return


#########################
##### RWA extension #####
#########################
#The code below contains RK4rwa and rate_eq_rwa for rotating wave approximation


@jit(nopython=True, nogil=True)
def RK4rwa(steps,stepsize,num,vibronic_trans,vibrational_trans,E_time,gamma,Gammabar,mu,f,f_0,drho_dt_empty,rho):
    for i in range(0,steps-2,2):
        i_new = int(i/2)
        k_1 = stepsize*rate_eq_rwa(num, vibronic_trans, vibrational_trans, rho[:,:,i_new], E_time[i], gamma, Gammabar, mu, f, f_0, drho_dt_empty)
        k_2 = stepsize*rate_eq_rwa(num, vibronic_trans, vibrational_trans, rho[:,:,i_new]+0.5*k_1, E_time[i+1], gamma, Gammabar, mu, f, f_0, drho_dt_empty)
        k_3 = stepsize*rate_eq_rwa(num, vibronic_trans, vibrational_trans, rho[:,:,i_new]+0.5*k_2, E_time[i+1], gamma, Gammabar, mu, f, f_0, drho_dt_empty)
        k_4 = stepsize*rate_eq_rwa(num, vibronic_trans, vibrational_trans, rho[:,:,i_new]+k_3, E_time[i+2], gamma, Gammabar, mu, f, f_0, drho_dt_empty)
        rho[:,:,i_new+1] = rho[:,:,i_new]+(k_1 + 2*k_2 + 2*k_3 + k_4)/6
    return rho

@jit(nopython=True, nogil=True)
def rate_eq_rwa(num,vibronic_trans,vibrational_trans,rho,E,gamma,Gammabar,mu,f,f_0,drho_dt):
    for i in range(num):
        for j in range(num):
            #drho_dt[0,0] is calculated last and is skipped at this point:
            if (i,j) == (0,0):
                pass
            #calculate diagonal elements (drho_dt[i,j] for i=j)
            elif i == j:
                sum1 =  -gamma[i,j]*rho[i,j]
                for k in range(i+1,num):
                    sum1 += gamma[i,k]*rho[k,k]
                sum2 = 0+0j
                for k in range(i):
                    if mu[k,i] == 0:
                        pass
                    else:
                        sum2 += mu[k,i]*(rho[k,i]*E-rho[i,k]*np.conj(E))
                for k in range(i+1,num):
                    if mu[i,k] == 0:
                        pass
                    else:
                        sum2 += mu[i,k]*(rho[k,i]*np.conj(E)-rho[i,k]*E)
                drho_dt[i,j] = sum1+1j/(2*hbar)*sum2
            #calculate non diagonal elements of vibronic transitions (drho_dt[i,j] for i<j):
            elif vibronic_trans[i,j] == 1:
                sum1 = -Gammabar[i,j]*rho[i,j] + 1j*(f[j,i]-f_0)*rho[i,j]
                sum2 = 0+0j
                if mu[i,j] == 0:
                    pass
                else:
                    sum2 += mu[i,j]*(rho[j,j]-rho[i,i])
                for k in range(num):
                    if k == j or k == i or mu[k,i] == 0:
                        pass
                    else:
                        sum2 += mu[k,i]*rho[k,j]
                for k in range(num):
                    if k == i or k == j or mu[k,j] == 0:
                        pass
                    else:
                        sum2 += -mu[k,j]*rho[i,k]
                drho_dt[i,j] = sum1+1j/(2*hbar)*np.conj(E)*sum2
            #calculate non diagonal elements of vibrational transitions (drho_dt[i,j] for i<j):
            elif vibrational_trans[i,j] == 1:
                sum1 = -Gammabar[i,j]*rho[i,j] + 1j*f[j,i]*rho[i,j]
                sum2 = 0+0j
                for k in range(i):
                    if k == j or mu[k,i] == 0:
                        pass
                    else:
                        sum2 += mu[k,i]*rho[k,j]*E
                for k in range(i+1,num):
                    if k == j or mu[i,k] == 0:
                        pass
                    else:
                        sum2 += mu[i,k]*rho[k,j]*np.conj(E)
                for k in range(j):
                    if k == i or mu[k,j] == 0:
                        pass
                    else:
                        sum2 += -mu[k,j]*rho[i,k]*np.conj(E)
                for k in range(j+1,num):
                    if k == i or mu[j,k] == 0:
                        pass
                    else:
                        sum2 += -mu[j,k]*rho[i,k]*E
                drho_dt[i,j] = sum1+1j/(2*hbar)*sum2
            #calculate non diagonal elements (drho_dt[i,j] for i>j):
            else:
                drho_dt[i,j] = np.conj(drho_dt[j,i])
    #calculate drho_dt[0,0]:
    sum1 = 0+0j
    for i in range(1,num):
        sum1 += drho_dt[i,i]
    drho_dt[0,0] = -sum1
    return drho_dt
