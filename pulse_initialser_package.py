import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from scipy.constants import c, pi, h, hbar

class pulse:
    '''Properties defined here:
    num : 1
        Number of pulses
    f_0 : float
        Central frequency of spectrum (rad*s-1)
        
    dt : float
        Pulse duration (FWHM) of transform limited pulse (s)
    E_0 : float
        Amplitude of transform limited pulse (V*m-1)
    phase_type : string
        Type of spectral phase transfer function, which is applied on the transform limited pulse
        Possible values:
        sinus (a*sin(b*x+c))
        taylor (a*x_0+b*x_0(x-x_0)+1/2*c*x_0(x-x_0)^2+1/6*d*x_0(x-x_0))
        sinus+taylor (a*sin(b*x+c)+d*x_0+e*x_0(x-x_0)+1/2*f*x_0(x-x_0)^2+1/6*g*x_0(x-x_0))
    phase_param : list of floats
        Parameters (a;b;c;...) for spectral phase transfer function
    t_limit : float
        Maximum time value (s)
        The pulse is simulated in the time period from -t_limit to t_limit.
    f_max : float
        Highest frequency occurring in the system (rad*s-1)
    freq : 1D numpy array
        Frequency grid (rad*s-1)
    phase_freq : 1D numpy array
        Spectral phase (rad)
    E_freq : 1D numpy array
        Spectrum (V*m-1)
    time : 1D numpy array
        Time grid (s)
    phase_time : 1D numpy array
        Time dependent temporal phase (rad)
    E_time : 1D numpy array
        Pulse (V*m-1)
    '''
    def __init__(self, f_0, dt, E_0, phase_type, phase_param, t_limit, f_max):
        #assign all variables of pulse class:
        self.f_0 = f_0 #central frequency of spectrum (rad s-1)
        self.dt = dt #pulse duration (FWHM) (s)
        self.E_0 = E_0 #amplitude (V m-1)
        self.phase_type = phase_type #type of spectral phase transfer function
        self.phase_param = phase_param #list of parameters for spectral phase transfer function
        self.t_limit = t_limit #maximum time value (s) 
        self.f_max = f_max #highest frequency occurring in the system (rad s-1)

        self.num = 1 #number of pulses
        self._rwa = False #this class does not use rotating wave approximation

        self.freq = None #frequency grid
        self.phase_freq = None #spectral phase
        self.E_freq = None #spectrum
        
        self.time = None #time grid
        self.phase_time = None #temporal phase
        self.E_time = None #pulse
    
    @classmethod
    def get(cls, input_file):
        '''Create pulse object with an input file.
        
        Variables:
        ----------
        input_file : string
            Name of input file.
            If the input file is located in a different file path, this path must also be specified.
        '''    
        #read data from input file into input dictionary:
        input_dict = {}
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    input_dict[line.split()[0]] = line.split()[1]
                except:
                    pass
        #change data type of some items in the input dictionary:
        input_dict['f_0'] = 2*pi*c/float(input_dict['f_0']) #central frequency of spectrum (converted from wavelength (m) to angular frequency (rad s-1))
        input_dict['dt'] = float(input_dict['dt'])
        input_dict['E_0'] = float(input_dict['E_0'])
        input_dict['phase_param'] = list(map(float, input_dict['phase_param'].split(';')))
        input_dict['t_limit'] = float(input_dict['t_limit'])

        #determine the highest frequency occurring in the system:
        f_max_1 = input_dict['f_0']
        f_max_2 = 0
        try:
            energy = list(map(float, input_dict['energy'].split(';')))
            f_max_2 = (energy[-1]-energy[0])*100*c*2*pi
        except:
            print('An attempt was made to determine the highest frequency occurring in the system.\nIt seems that no system parameters are given in the input file.\nTherefore the frequency of the laser pulse was assumed to be the highest occurring frequency.')
        f_max = max([f_max_1,f_max_2])
        return cls(input_dict['f_0'],input_dict['dt'],input_dict['E_0'],input_dict['phase_type'],input_dict['phase_param'],input_dict['t_limit'],f_max)

   
    def spectral(self):
        '''Calculate spectral properties of pulse object.
        (frequency grid (freq), spectral phase (phase_freq), spectrum (E_freq))
        '''
        #generate frequency grid (self.freq):
        ###calculate stepsize of frequency grid (f_step):
        f_step = 2*pi/(2*self.t_limit)
        ###estimate maximum frequency value (f_limit):
        ###f_limit is chosen so that the frequency grid fits the 5 sigma area of the gaussian spectrum (self.f_0+5*sigma) but it is at least 20 times the highest frequency of the system (20*self.f_max).              
        sigma = 2*np.sqrt(np.log(2))/self.dt
        f_limit = max((20*self.f_max, self.f_0+5*sigma)) #maximum frequency value (rad s-1)
        ###calculate size of frequency grid (N): 
        N = int(f_limit/f_step)
        ###generate frequency grid:
        self.freq = np.linspace(0,f_limit,num=N+1)
        
        #calculate spectral phase (self.phase_freq):
        if self.phase_type == 'sinus+taylor':
            self.phase_freq = self.phase_param[0]*np.sin(self.phase_param[1]*(self.freq-self.f_0)+self.phase_param[2]) + self.phase_param[3]+self.phase_param[4]*(self.freq-self.f_0)+1/2*self.phase_param[5]*np.power(self.freq-self.f_0, 2)+1/6*self.phase_param[6]*np.power(self.freq-self.f_0, 3)
        elif self.phase_type == 'sinus':
            self.phase_freq = self.phase_param[0]*np.sin(self.phase_param[1]*(self.freq-self.f_0)+self.phase_param[2])
        elif self.phase_type == 'sinus_tophat':
            self.phase_freq = self.phase_param[0]*np.sin(self.phase_param[1]*(self.freq-self.f_0)+self.phase_param[2])
        elif self.phase_type == 'taylor':
            self.phase_freq = self.phase_param[0]+self.phase_param[1]*(self.freq-self.f_0)+1/2*self.phase_param[2]*np.power(self.freq-self.f_0, 2)+1/6*self.phase_param[3]*np.power(self.freq-self.f_0, 3)
        else:
            print('No valid phase type detected. Spectral phase set to 0.\nAccepted phase types: sinus, taylor, sinus+taylor')
            self.phase_freq = np.zeros(N+1)
            
        #generate gaussian spectrum:
        gaussian_freq = self.E_0*self.dt*np.sqrt(pi/2/np.log(2))*np.exp(-np.power(self.dt*(self.freq - self.f_0), 2)/8/np.log(2))
        
        #calculate spectrum (self.E_freq):
        self.E_freq = gaussian_freq*np.exp(-1j*self.phase_freq)
        print('Calculation of spectral parameters finished.')
        return
            
            
    def temporal(self):
        '''Calculate temporal properties of pulse object.
        (time grid (time), pulse (E_time)).
        The time dependent temporal phase (phase_time) is calculated with temporal_phase().
        '''
        #calculate pulse:
        ###zero padding:
        N = self.freq.shape[0]
        E_freq_zero = np.zeros(2*N-1, dtype=complex)
        E_freq_zero[:N] = self.E_freq
        ###inverse fast fourier transform (ifft):
        self.E_time = np.real(scipy.fft.fftshift(scipy.fft.ifft(E_freq_zero)))*(2*self.freq[-1])/(2*pi)
        
        #generate time grid:
        self.time = scipy.fft.fftshift(scipy.fft.fftfreq(self.E_time.shape[0],(self.freq[1]-self.freq[0])/(2*pi)))

        print('Calculation of temporal parameters finished.')
        return


    def temporal_phase(self):
        '''Calculate temporal phase (phase_time) of pulse object.
        '''
        N = self.freq.shape[0]
        E_freq_zero = np.zeros(2*N-1, dtype=complex)
        E_freq_zero[:N] = self.E_freq
        E_time_complex = scipy.fft.fftshift(scipy.fft.ifft(E_freq_zero))*(2*self.freq[-1])/(2*pi)
        self.phase_time = np.angle(E_time_complex)
        #self.phase_time = (np.angle(E_time_complex)-self.f_0*self.time + pi)%(2*pi) - pi
        #self.phase_time = (np.unwrap(np.angle(E_time_complex))-self.f_0*self.time + pi)%(2*pi) - pi

        print('Calculation of time dependent part of temporal phase finished.')
        return


    def plot(self):
        '''Plot spectral and temporal properties of pulse object.
        '''
        fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(8,6), sharex='row')
        ax[0][0].set_xlabel('frequency (rad s-1)')
        ax[0][0].set_ylabel('E (V/m)')
        ax[0][0].set_title('spectrum')
        ax[0][0].plot(self.freq, np.real(self.E_freq), label='real')
        ax[0][0].plot(self.freq, np.imag(self.E_freq), label='imag')
        ax[0][0].legend(loc='best')

        ax[0][1].set_xlabel('frequency (rad s-1)')
        ax[0][1].set_ylabel('phase (rad)')
        ax[0][1].set_title('spectral phase')
        ax[0][1].plot(self.freq, self.phase_freq, label='phase')
        
        ax[1][0].set_xlabel('time (s)')
        ax[1][0].set_ylabel('E (V/m)')
        ax[1][0].set_title('pulse')
        ax[1][0].plot(self.time, np.real(self.E_time), label='real')
        ax[1][0].plot(self.time, np.imag(self.E_time), label='imag')
        ax[1][0].legend(loc='best')

        
        ax[1][1].set_xlabel('time (s)')
        ax[1][1].set_ylabel('phase (rad)')
        ax[1][1].set_title('temporal phase')
        try:
            ax[1][1].plot(self.time, self.phase_time, label='phase')
        except:
            pass

        fig.subplots_adjust(hspace=0.5, wspace = 0.4)
        plt.show()
        return




class npulse:
    '''Properties defined here:
    num : int
        Number of pulses
    f_0 : list of floats
        List of central frequencies of spectra (rad*s-1)
        
    dt : list of floats
        List of pulse durations (FWHM) of transform limited pulses (s)
    E_0 : list of floats
        List of amplitudes of transform limited pulses (V*m-1)
    phase_type : list of strings
        List of types of spectral phase transfer functions, which are applied on the transform limited pulses
        Possible values:
        sinus (a*sin(b*x+c))
        taylor (a*x_0+b*x_0(x-x_0)+1/2*c*x_0(x-x_0)^2+1/6*d*x_0(x-x_0))
        sinus+taylor (a*sin(b*x+c)+d*x_0+e*x_0(x-x_0)+1/2*f*x_0(x-x_0)^2+1/6*g*x_0(x-x_0))
    phase_param : list of lists of floats
        List of parameters (a;b;c;...) for spectral phase transfer functions
    t_limit : float
        Maximum time value (s)
        The pulse is simulated in the time period from -t_limit to t_limit.
    f_max : float
        Highest frequency occurring in the system (rad*s-1)
    freq : 1D numpy array
        Frequency grid (rad*s-1)
    phase_freq : 2D numpy array
        Spectral phases (rad)
    E_freq : 2D numpy array
        Spectra (V*m-1)
    time : 1D numpy array
        Time grid (s)
    phase_time : 1D numpy array
        Time dependent temporal phase (rad)
    E_time : 1D numpy array
        Pulse (V*m-1)
    '''
    



    def __init__(self, list_of_pulses):
        #assign all variables of pulse class:
        self.num = len(list_of_pulses) #number of pulses
        self.f_0 = [list_of_pulses[i].f_0 for i in range(self.num)] #list of central frequencies of spectra (rad s-1)
        self.dt = [list_of_pulses[i].dt for i in range(self.num)] #list of pulse durations (FWHM) (s)
        self.E_0 = [list_of_pulses[i].E_0 for i in range(self.num)] #list of amplitudes (V m-1)
        self.phase_type = [list_of_pulses[i].phase_type for i in range(self.num)] #list of types of spectral phase transfer functions
        self.phase_param = [list_of_pulses[i].phase_param for i in range(self.num)] #list of lists of parameters for spectral phase transfer functions
        self.t_limit = max([list_of_pulses[i].t_limit for i in range(self.num)]) #maximum time value (s) 
        self.f_max = max([list_of_pulses[i].f_max for i in range(self.num)]) #highest frequency occurring in the system (rad s-1)

        self._rwa = False #this class does not use rotating wave approximation

        self.freq = 0 #frequency grid
        self.phase_freq = 0 #spectral phase
        self.E_freq = 0 #spectrum
        
        self.time = 0 #time grid
        self.phase_time = 0 #temporal phase
        self.E_time = 0 #pulse
    
    @classmethod
    def get(cls, input_file):
        '''Create npulse object with an input file.
        
        Variables:
        ----------
        input_file : string
            Name of input file.
            If the input file is located in a different file path, this path must also be specified.
        '''   
        #read data from input file into input dictionary:
        input_dict = {}
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    input_dict[line.split()[0]] = line.split()[1:]
                except:
                    pass
        #process general parameters:
        input_dict['t_limit'] = float(input_dict['t_limit'][0])
        try:
            input_dict['num'] = int(input_dict['num'][0])
            num = input_dict['num']
        except:
            num = 1
            print('Parameter "num" (number of pulses) has not been detekted. Therefore "num" is assumed to be 1.')
        #process pulse parameters:
        input_dict['f_0'] = input_dict['f_0'][0:num]
        input_dict['dt'] = input_dict['dt'][0:num]
        input_dict['E_0'] =  input_dict['E_0'][0:num]
        input_dict['phase_type'] = input_dict['phase_type'][0:num]
        input_dict['phase_param'] = input_dict['phase_param'][0:num]
        #change data type of some items in the input dictionary:
        for i in range(num):
            input_dict['f_0'][i] = 2*pi*c/float(input_dict['f_0'][i]) #list of central frequencies of spectra (converted from wavelength (m) to angular frequency (rad s-1))
            input_dict['dt'][i] = float(input_dict['dt'][i])
            input_dict['E_0'][i] = float(input_dict['E_0'][i])
            input_dict['phase_param'][i] = list(map(float, input_dict['phase_param'][i].split(';')))
            
        #determine the highest frequency occurring in the system:
        f_max_1 = max(input_dict['f_0'])
        f_max_2 = 0
        try:
            energy = list(map(float, input_dict['energy'][0].split(';')))
            f_max_2 = (energy[-1]-energy[0])*100*c*2*pi
        except:
            print('An attempt was made to determine the highest frequency occurring in the system.\nIt seems that no system parameters are given in the input file.\nTherefore the frequency of the laser pulse was assumed to be the highest occurring frequency.')
        f_max = max([f_max_1,f_max_2])

        list_of_pulses = [pulse(input_dict['f_0'][i],input_dict['dt'][i],input_dict['E_0'][i],input_dict['phase_type'][i],input_dict['phase_param'][i],input_dict['t_limit'],f_max) for i in range(num)]
        for i in range(num):
            print(list_of_pulses[i])
        return cls(list_of_pulses)


    def spectral(self):
        '''Calculate spectral properties of npulse object.
        (frequency grid (freq), spectral phase (phase_freq), spectrum (E_freq))
        '''
        #generate frequency grid (self.freq):
        ###calculate stepsize of frequency grid (f_step):
        f_step = 2*pi/(2*self.t_limit)
        ###estimate maximum frequency value
        ###f_limit is chosen so that the frequency grid fits the 5 sigma area of the gaussian spectrum (self.f_0+5*sigma) but it is at least 20 times the highest frequency of the system (20*self.f_max).
        sigma = [2*np.sqrt(np.log(2))/self.dt[i] for i in range(self.num)]
        f_limit = max([20*self.f_max] + [self.f_0[i]+5*sigma[i] for i in range(self.num)])
        ###calculate size of frequency grid (N): 
        N = int(f_limit/f_step)
        ###generate frequency grid:
        self.freq = np.linspace(0,f_limit,num=N+1)
        
        #calculate spectral phase (self.phase_freq):
        self.phase_freq = np.empty((self.num,N+1))
        for i in range(self.num):    
            if self.phase_type[i] == 'sinus+taylor':
                self.phase_freq[i,:] = self.phase_param[i][0]*np.sin(self.phase_param[i][1]*(self.freq-self.f_0[i])+self.phase_param[i][2])+self.phase_param[i][3]+self.phase_param[i][4]*(self.freq-self.f_0[i])+1/2*self.phase_param[i][5]*np.power(self.freq-self.f_0[i], 2)+1/6*self.phase_param[i][6]*np.power(self.freq-self.f_0[i], 3)
            elif self.phase_type[i] == 'sinus':
                self.phase_freq[i,:] = self.phase_param[i][0]*np.sin(self.phase_param[i][1]*(self.freq-self.f_0[i])+self.phase_param[i][2])
            elif self.phase_type[i] == 'sinus+heaviside':
                self.phase_freq[i,:] = self.phase_param[i][0]*np.sin(self.phase_param[i][1]*(self.freq-self.f_0[i])+self.phase_param[i][2])
            elif self.phase_type[i] == 'taylor':
                self.phase_freq[i,:] = self.phase_param[i][0]+self.phase_param[i][1]*(self.freq-self.f_0[i])+1/2*self.phase_param[i][2]*np.power(self.freq-self.f_0[i], 2)+1/6*self.phase_param[i][3]*np.power(self.freq-self.f_0[i], 3)
            else:
                print('No valid phase type detected. Spectral phase set to 0.\nAccepted phase types: sinus, taylor, sinus+taylor')
                self.phase_freq[i,:] = np.zeros(N+1)
            
        #generate gaussian spectrum:
        gaussian_freq = np.empty((self.num,N+1), dtype=complex)
        for i in range(self.num):
            gaussian_freq[i,:] = self.E_0[i]*self.dt[i]*np.sqrt(pi/2/np.log(2))*np.exp(-np.power(self.dt[i]*(self.freq - self.f_0[i]), 2)/8/np.log(2))
        
        #calculate spectrum (self.E_freq):
        self.E_freq = np.empty((self.num,N+1), dtype=complex)
        for i in range(self.num):
            self.E_freq[i,:] = gaussian_freq[i,:]*np.exp(-1j*self.phase_freq[i,:])
        
            print('Calculation of spectral parameters finished.')
        return
            
            
    def temporal(self):
        '''Calculate temporal properties of npulse object.
        (time grid (time), pulse (E_time)).
        The time dependent temporal phase (phase_time) is calculated with temporal_phase().
        '''
        #calculate pulse:
        ###zero padding:
        N = self.freq.shape[0]
        E_freq_zero = np.zeros((self.num,2*N-1), dtype=complex)
        E_freq_zero[:,:N] = self.E_freq
        ###inverse fast fourier transform (ifft):
        self.E_time = np.zeros(2*N-1)
        for i in range(self.num):
            self.E_time = self.E_time + np.real(scipy.fft.fftshift(scipy.fft.ifft(E_freq_zero[i,:])))*(2*self.freq[-1])/(2*pi)
        print("Pulse Type:", self.phase_type[0])
        #generate time grid:
        self.time = scipy.fft.fftshift(scipy.fft.fftfreq(self.E_time.shape[0],(self.freq[1]-self.freq[0])/(2*pi)))
        #print(self.phase_type[i])
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]
       
        if self.phase_type[i] == 'sinus+heaviside':
               print(self.phase_param[0][3])
               val_1 = find_nearest(self.time, self.phase_param[0][3])
               idx = np.where(self.time== val_1)
               print(idx)
               print(np.shape(self.E_time))
               self.E_time[0:int(idx[0])]= 0 
               print(self.E_time)
               
        else:
            #print('no heaviside')
            
            print('Calculation of temporal parameters finished.')
        return


    def temporal_phase(self):
        '''Calculate temporal phase (phase_time) of npulse object.
        '''
        N = self.freq.shape[0]
        E_freq_zero = np.zeros((self.num,2*N-1), dtype=complex)
        E_freq_zero[:,:N] = self.E_freq
        E_time_complex = np.empty(2*N-1, dtype=complex)
        for i in range(self.num):
            E_time_complex = E_time_complex + scipy.fft.fftshift(scipy.fft.ifft(E_freq_zero[i,:]))*(2*self.freq[-1])/(2*pi)
        self.phase_time = np.angle(E_time_complex)
        #self.phase_time = (np.angle(E_time_complex)-self.f_0[0]*self.time + pi)%(2*pi) - pi
        #self.phase_time = (np.unwrap(np.angle(E_time_complex))-self.f_0[0]*self.time + pi)%(2*pi) - pi

        print('Calculation of time dependent part of temporal phase finished.')
        return
    

    def plot(self):
        '''Plot spectral and temporal properties of npulse object.
        '''
        fig, ax = plt.subplots(nrows=1+self.num, ncols=2,figsize=(8,3*(1+self.num)), sharex='row')
        for i in range(self.num):
            ax[i][0].set_xlabel('frequency (rad s-1)')
            ax[i][0].set_ylabel('E (V/m)')
            ax[i][0].set_title(str('spectrum of pulse '+str(i+1)))
            ax[i][0].plot(self.freq, np.real(self.E_freq[i,:]), label='real')
            ax[i][0].plot(self.freq, np.imag(self.E_freq[i,:]), label='imag')
            ax[i][0].legend(loc='best')

            ax[i][1].set_xlabel('frequency (rad s-1)')
            ax[i][1].set_ylabel('phase (rad)')
            ax[i][1].set_title(str('spectral phase of pulse '+str(i+1)))
            ax[i][1].plot(self.freq, self.phase_freq[i,:], label='phase')
            
        ax[self.num][0].set_xlabel('time (s)')
        ax[self.num][0].set_ylabel('E (V/m)')
        ax[self.num][0].set_title('pulse')
        ax[self.num][0].plot(self.time, np.real(self.E_time), label='real')
        ax[self.num][0].plot(self.time, np.imag(self.E_time), label='imag')
        ax[self.num][0].legend(loc='best')
        
        ax[self.num][1].set_xlabel('time (s)')
        ax[self.num][1].set_ylabel('phase (rad)')
        ax[self.num][1].set_title('temporal phase')
        
        try:
            ax[self.num][1].plot(self.time, self.phase_time, label='phase')
        except:
            pass
            
        fig.subplots_adjust(hspace=0.5, wspace = 0.4)
        plt.show()
        
        return



#########################
##### RWA extension #####
#########################
#The code below contains pulserwa and npulserwa class for rotating wave approximation


class pulserwa(pulse):
    def __init__(self, f_0, dt, E_0, phase_type, phase_param, t_limit, f_max):
        super().__init__(f_0, dt, E_0, phase_type, phase_param, t_limit, f_max)
        self._rwa = True #this class does use rotating wave approximation

    def spectral(self):
        '''Calculate spectral properties of pulserwa object.
        (frequency grid (freq), spectral phase (phase_freq), spectrum (E_freq))
        '''
        #generate frequency grid (self.freq):
        ###calculate stepsize of frequency grid (f_step):
        f_step = 2*pi/(2*self.t_limit)
        ###estimate maximum frequency value (f_limit):
        ###f_limit is chosen so that the frequency grid fits the 5 sigma area of the gaussian spectrum (self.f_0+5*sigma) but it is at least 20 times the highest frequency of the system (20*self.f_max).
        sigma = 2*np.sqrt(np.log(2))/self.dt
        f_limit = max((20*self.f_max,self.f_0+5*sigma))
        ###calculate size of frequency grid (N): 
        N = 2*int(f_limit/f_step)
        ###generate frequency grid:
        self.freq = np.linspace(-f_limit,f_limit,num=N+1)
        
        #calculate spectral phase (self.phase_freq):
        if self.phase_type == 'sinus+taylor':
            self.phase_freq = self.phase_param[0]*np.sin(self.phase_param[1]*self.freq+self.phase_param[2])+self.phase_param[3]+self.phase_param[4]*self.freq+1/2*self.phase_param[5]*np.power(self.freq, 2)+1/6*self.phase_param[6]*np.power(self.freq, 3)
        elif self.phase_type == 'sinus':
            self.phase_freq = self.phase_param[0]*np.sin(self.phase_param[1]*self.freq+self.phase_param[2])
        elif self.phase_type == 'sinus_tophat':
            self.phase_freq = self.phase_param[0]*np.sin(self.phase_param[1]*self.freq+self.phase_param[2])
        elif self.phase_type == 'taylor':
            self.phase_freq = self.phase_param[0]+self.phase_param[1]*self.freq+1/2*self.phase_param[2]*np.power(self.freq, 2)+1/6*self.phase_param[3]*np.power(self.freq, 3)
        else:
            print('No valid phase type detected. Spectral phase set to 0.\nAccepted phase types: sinus, taylor, sinus+taylor')
            self.phase_freq = np.zeros(N+1)
            
        #generate gaussian spectrum:
        gaussian_freq = self.E_0*self.dt*np.sqrt(pi/2/np.log(2))*np.exp(-np.power(self.dt*self.freq, 2)/8/np.log(2))
        

        #calculate spectrum (self.E_freq):
        self.E_freq = gaussian_freq*np.exp(-1j*self.phase_freq)
        print('Calculation of spectral properties finished.')
        return
    
    def temporal(self):
        '''Calculate temporal properties of pulserwa object.
        (time grid (time), pulse (E_time)).
        The time dependent temporal phase (phase_time) is calculated with temporal_phase().
        '''
        #calculate pulse:
        ###inverse fast fourier transform (ifft):
        self.E_time = scipy.fft.fftshift(scipy.fft.ifft(scipy.fft.ifftshift(self.E_freq)))*(self.freq[-1]-self.freq[0])/(2*pi)
        
        #generate time grid:
        self.time = scipy.fft.fftshift(scipy.fft.fftfreq(self.E_time.shape[0],(self.freq[1]-self.freq[0])/(2*pi)))

        print('Calculation of temporal properties finished.')
        return
    
    def temporal_phase(self):
        '''Calculate temporal phase (phase_time) of pulserwa object.
        '''
        self.phase_time = np.angle(self.E_time)

        print('Calculation of time dependent part of temporal phase finished.')
        return

class npulserwa(npulse):
    def __init__(self, list_of_pulses):
        super().__init__(list_of_pulses)
        self._rwa = True #rotating wave approximation

    def spectral(self):
        '''Calculate spectral properties of npulserwa object.
        (frequency grid (freq), spectral phase (phase_freq), spectrum (E_freq))
        '''
        #generate frequency grid (self.freq):
        ###calculate stepsize of frequency grid (f_step):
        f_step = 2*pi/(2*self.t_limit)
        ###estimate maximum frequency value (f_limit):
        ###f_limit is chosen so that the frequency grid fits the 5 sigma area of the gaussian spectrum (self.f_0+5*sigma) but it is at least 20 times the highest frequency of the system (20*self.f_max).
        sigma = [2*np.sqrt(np.log(2))/self.dt[i] for i in range(self.num)]
        f_limit = max([20*self.f_max] + [self.f_0[i]+5*sigma[i] for i in range(self.num)])
        ###calculate size of frequency grid (N):
        N = 2*int(f_limit/f_step)
        ###generate frequency grid:
        self.freq = np.linspace(-f_limit,f_limit,num=N+1)
        
        #calculate spectral phase (self.phase_freq):
        self.phase_freq = np.empty((self.num,N+1))
        for i in range(self.num):    
            if self.phase_type[i] == 'sinus+taylor':
                self.phase_freq[i,:] = self.phase_param[i][0]*np.sin(self.phase_param[i][1]*self.freq+self.phase_param[i][2])+self.phase_param[i][3]+self.phase_param[i][4]*self.freq+1/2*self.phase_param[i][5]*np.power(self.freq, 2)+1/6*self.phase_param[i][6]*np.power(self.freq, 3)
            elif self.phase_type[i] == 'sinus':
                self.phase_freq[i,:] = self.phase_param[i][0]*np.sin(self.phase_param[i][1]*self.freq+self.phase_param[i][2])
            elif self.phase_type[i] == 'taylor':
                self.phase_freq[i,:] = self.phase_param[i][0]+self.phase_param[i][1]*self.freq+1/2*self.phase_param[i][2]*np.power(self.freq, 2)+1/6*self.phase_param[i][3]*np.power(self.freq, 3)
            elif self.phase_type[i] == 'sinus_tophat':
                self.phase_freq[i,:] = self.phase_param[i][0]*np.sin(self.phase_param[i][1]*self.freq+self.phase_param[i][2])
            else:
                print('No valid phase type detected. Spectral phase set to 0.\nAccepted phase types: sinus, taylor, sinus+taylor')
                self.phase_freq[i,:] = np.zeros(N+1)
            
        #generate gaussian spectrum:
        gaussian_freq = np.empty((self.num,N+1), dtype=complex)
        for i in range(self.num):
            gaussian_freq[i,:] = self.E_0[i]*self.dt[i]*np.sqrt(pi/2/np.log(2))*np.exp(-np.power(self.dt[i]*self.freq, 2)/8/np.log(2))
        
        #calculate spectrum (self.E_freq):
        self.E_freq = np.empty((self.num,N+1), dtype=complex)
        for i in range(self.num):
            self.E_freq[i,:] = gaussian_freq[i,:]*np.exp(-1j*self.phase_freq[i,:])
        
        print('Calculation of spectral properties finished.')
        return

    def temporal(self):
        '''Calculate temporal properties of npulserwa object.
        (time grid (time), pulse (E_time)).
        The time dependent temporal phase (phase_time) is calculated with temporal_phase().
        '''
        #calculate pulse:
        ###inverse fast fourier transform (ifft):
        self.E_time = np.zeros(self.freq.shape[0], dtype=complex)
        for i in range(self.num):
            self.E_time = self.E_time + scipy.fft.fftshift(scipy.fft.ifft(scipy.fft.ifftshift(self.E_freq[i,:])))*(self.freq[-1]-self.freq[0])/(2*pi)

        #generate time grid:
        self.time = scipy.fft.fftshift(scipy.fft.fftfreq(self.E_time.shape[0],(self.freq[1]-self.freq[0])/(2*pi)))

        print('Calculation of temporal properties finished.')
        return
        
    def temporal_phase(self):
        '''Calculate temporal phase (phase_time) of npulserwa object.
        '''
        self.phase_time = np.angle(self.E_time)

        print('Calculation of time dependent part of temporal phase finished.')
        return
    