import numpy as np
from scipy.constants import c, pi, h, hbar 
import scipy.constants as sc
from scipy.stats import boltzmann
class system:
    '''Properties defined here:

    size : list of int
        Size parameters of system
        Example: size = [2,3,4]: The system has 3 electronic niveaus,
        each divided in 2, 3, and 4 vibrational niveaus respectively.

    num : int
        Number of energy states
    
    gamma : 2D numpy array
        Population decay rates (s-1)

    Gamma : 2D numpy array
        Dephasing rates (s-1)

    Gammabar : 2D numpy array
        Coherence decay rates (s-1)

    f : 2D numpy array
        Transition frequencies (rad*s-1)

    mu : 2D numpy array
        Dipole coupling coefficients of transitions (C*m)

    energy : list of floats
        Energies of energy states (cm-1)

    rho_init : list of floats
        Inital population values

    vibronic_trans : 2D numpy array
        Binary matrix indicating vibronic transitions

    vibrational_trans : 2D numpy array
        Binary matrix indicating vibrational transitions
    '''
    def __init__(self, size, gamma, Gamma, mu, energy, rho_init,nbar,delta):
        #assign all variables of system class:
        self.size = size #list with size parameters of system
        self.gamma = gamma #2D numpy array with decay rates (s-1)
        self.Gamma = Gamma #2D numpy array with coherence dephasing rates (s-1)
        self.mu = mu #2D numpy array with effective dipole coupling coefficients (C m)
        self.energy = energy #list with energies of energy states
        self.rho_init = rho_init #list with initial population values

        self.num = sum(size) #number of energy states
        self.nbar = nbar
        self.delta = delta
        self._rwa = False

        #calculate binary matrices of vibronic and vibrational transitions:
        ### this binary matrices serve as a tool for the generation of the rate equations
        vibronic_trans = np.zeros((sum(size),sum(size)))
        vibrational_trans = np.zeros((sum(size),sum(size)))
        for i in range(sum(size)):
            for j in range(i+1,sum(size)):
                vibronic_trans[i,j] = 1
        for i in range(len(size)):
            for j in range(size[i]):
                for k in range(j+1,size[i]):
                    vibrational_trans[j+sum(size[0:i]),k+sum(size[0:i])] = 1
                    vibronic_trans[j+sum(size[0:i]),k+sum(size[0:i])] = 0
        self.vibronic_trans = vibronic_trans #binary matrix of all vibronic transitions
        self.vibrational_trans = vibrational_trans #binary matrix of all vibrational transitions

        #calculate 2D numpy array with transition frequencies (rad s-1):
        f = np.zeros((sum(size),sum(size)))
        for i in range(sum(size)):
            for j in range(sum(size)):
                f[i,j] = abs((energy[j]-energy[i])*100*c*2*pi)
        self.f = f #2D numpy array with transition frequencies (rad s-1)

        #calculate 2D numpy array with coherence decay rates (s-1) 
        Gammabar = np.zeros((sum(size),sum(size)))
        for i in range(sum(size)):
            for j in range(i+1,sum(size)):
                Gammabar[i,j] = (gamma[i,i]+gamma[j,j])/2+Gamma[i,j]
        self.Gammabar = Gammabar #2D numpy array with coherence decay rates (s-1) 

    @classmethod
    def get(cls, input_file):
        '''Create system object with a basic or extended input file
        
        Variables:
        ----------
        input_file : string
            Name of input file.
            If the input file is located in a different file path, this path must also be specified.
        '''
        try:
            my_system = cls.get_basic(input_file)
        except:
            try:
                my_system = cls.get_extended(input_file)
            except:
                print('Ooops. Something went wrong...\nThe file "{}" could neither be loaded as basic input file nor as extended input file.\nFor error messages try to load the file manually with system.get_basic or system.get_extended.'.format(input_file))
                return
        return my_system
        
    @classmethod
    def get_basic(cls, input_file):
        '''Create system object with a basic input file
        
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
        input_dict['size'] = list(map(int, input_dict['size'].split(';')))
        input_dict['gamma'] = float(input_dict['gamma'])
        input_dict['Gamma_ele'] = float(input_dict['Gamma_ele'])
        input_dict['Gamma_vib'] = float(input_dict['Gamma_vib'])
        input_dict['mu'] = float(input_dict['mu'])
        input_dict['energy'] = list(map(float, input_dict['energy'].split(';')))
        input_dict['rho_init'] = list(map(float, input_dict['rho_init'].split(';')))

        # calculate binary matrices of vibronic and vibrational transitions:
        vibronic_trans = np.zeros((sum(input_dict['size']),sum(input_dict['size'])))
        vibrational_trans = np.zeros((sum(input_dict['size']),sum(input_dict['size'])))
        for i in range(sum(input_dict['size'])):
            for j in range(i+1,sum(input_dict['size'])):
                vibronic_trans[i,j] = 1
        for i in range(len(input_dict['size'])):
            for j in range(input_dict['size'][i]):
                for k in range(j+1,input_dict['size'][i]):
                    vibrational_trans[j+sum(input_dict['size'][0:i]),k+sum(input_dict['size'][0:i])] = 1
                    vibronic_trans[j+sum(input_dict['size'][0:i]),k+sum(input_dict['size'][0:i])] = 0

        # generate gamma matrix:
        gamma = np.zeros((sum(input_dict['size']),sum(input_dict['size'])))
        for i in range(sum(input_dict['size'])):
            for j in range(i+1,sum(input_dict['size'])):
                gamma[i,j] = input_dict['gamma']
        for i in range(1,sum(input_dict['size'])):
            gamma[i,i] = sum(gamma[0:i,i])

        # generate Gamma matrix:
        Gamma = np.zeros((sum(input_dict['size']),sum(input_dict['size'])))
        for i in range(sum(input_dict['size'])):
            for j in range(i+1,sum(input_dict['size'])):
                if vibronic_trans[i,j] == 1:
                    Gamma[i,j] = input_dict['Gamma_ele']
                else:
                    Gamma[i,j] = input_dict['Gamma_vib']            

        # generate mu matrix:
        mu = np.zeros((sum(input_dict['size']),sum(input_dict['size'])))
        for i in range(sum(input_dict['size'])):
            for j in range(i+1,sum(input_dict['size'])):
                if vibronic_trans[i,j] == 1:
                    mu[i,j] = input_dict['mu']
                    mu[j,i] = input_dict['mu']

        # # consider parity if available:
        # try:
        #     input_dict['parity'] = list(map(str, input_dict['parity'].split(';')))
        #     ### generate list of lists of indices of energy states:
        #     index_list = []
        #     for i in range(len(input_dict['size'])):
        #         index_list.append([j+sum(input_dict['size'][0:i]) for j in range(input_dict['size'][i])])
        #     ### set mu for Laporte forbidden transitions to zero:
        #     for i in range(len(input_dict['size'])):
        #         for j in range(i+1,len(input_dict['size'])):
        #             if input_dict['parity'][i] == input_dict['parity'][j]:
        #                 for k in index_list[i]:
        #                     for l in index_list[j]:
        #                         mu[k,l] = 0
        #                         mu[l,k] = 0
        #             else:
        #                 pass
        #     print('Additional parity information has been detected and was taken into account.')
        # except:
        #     pass

        return cls(input_dict['size'],  gamma, Gamma, mu, input_dict['energy'], input_dict['rho_init'])

    @classmethod
    def get_extended(cls, input_file):
        '''Create system object with an extended input file
        
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
        input_dict['temp'] = float(input_dict['temp'])
        input_dict['delta'] = float(input_dict['delta'])
        input_dict['size'] = list(map(int, input_dict['size'].split(';')))
        input_dict['gamma'] = input_dict['gamma'].split('];[')
        input_dict['Gamma'] = input_dict['Gamma'].split('];[')
        input_dict['mu'] = input_dict['mu'].split('];[')
        for i in range(sum(input_dict['size'])):
            for j in ['gamma','Gamma','mu']:
                input_dict[j][i] = input_dict[j][i].replace('[','')
                input_dict[j][i] = input_dict[j][i].replace(']','')
                input_dict[j][i] = list(map(float, input_dict[j][i].split(';')))
        input_dict['energy'] = list(map(float, input_dict['energy'].split(';')))
        
        #Initial Populations with Boltzmann
        omega_res = (input_dict['energy'][1]-input_dict['energy'][0])*c*100
        if input_dict['temp'] ==0:
            input_dict['rho_init'] = list(map(float, input_dict['rho_init'].split(';')))
            input_dict['rho_init'][0]=1
            for i in range(sum(input_dict['size'])-1):
                input_dict['rho_init'][i+1]=0
            nbar = 0
        else:
            input_dict['rho_init'] = list(map(float, input_dict['rho_init'].split(';')))
            z= np.sum(np.exp(-np.asarray(input_dict['energy'])/(0.6950356 *input_dict['temp'])),dtype='f')
            input_dict['rho_init'] = np.exp(-np.asarray(input_dict['energy'])/(0.6950356*input_dict['temp']))/z
            nbar = np.exp((hbar*omega_res/(sc.Boltzmann*input_dict['temp']))-1)**(-1)
            
            #nt = np.arctanh(hbar*omega_res/(2*sc.Boltzmann*input_dict['temp'])
            #nbar = 0.5*((1/nt)-1)
        
        print("nbar",nbar)
        # calculate binary matrices of vibronic and vibrational transitions:
        vibronic_trans = np.zeros((sum(input_dict['size']),sum(input_dict['size'])))
        vibrational_trans = np.zeros((sum(input_dict['size']),sum(input_dict['size'])))
        for i in range(sum(input_dict['size'])):
            for j in range(i+1,sum(input_dict['size'])):
                vibronic_trans[i,j] = 1
        for i in range(len(input_dict['size'])):
            for j in range(input_dict['size'][i]):
                for k in range(j+1,input_dict['size'][i]):
                    vibrational_trans[j+sum(input_dict['size'][0:i]),k+sum(input_dict['size'][0:i])] = 1
                    vibronic_trans[j+sum(input_dict['size'][0:i]),k+sum(input_dict['size'][0:i])] = 0
        
        # generate gamma matrix:
        gamma = np.array(input_dict['gamma'])
        for i in range(1,sum(input_dict['size'])):
            gamma[i,i] = sum(gamma[0:i,i])

        # generate Gamma matrix:
        Gamma = np.array(input_dict['Gamma'])
        
        # generate mu matrix:
        mu = np.array(input_dict['mu'])
        for i in range(sum(input_dict['size'])):
            for j in range(i+1,sum(input_dict['size'])):
                    mu[j,i] = mu[i,j]
                    
        delta = np.array(input_dict['delta'])
        print("Starting Populations:",input_dict['rho_init'])
        print("Mu:",input_dict['mu'])
                    
        return cls(input_dict['size'],  gamma, Gamma, mu, input_dict['energy'], input_dict['rho_init'],nbar,delta)

    


    def print_rate_eq(self, string=False):
        '''Print density matrix rate equation.'''
        #Start writing rate equation:
        main_str = ''
        for i in range(self.num):
            for j in range(self.num):
                #generate drho_dt[0,0]:
                if (i,j) == (0,0):
                    part_str = 'd\u03C1/dt[0,0] ='
                    for k in range(1,self.num):
                        part_str += ' -d\u03C1/dt['+str(k)+','+str(k)+']'
                    part_str += '\n\n'
                    main_str += part_str
                #generate diagonal elements (drho_dt[i,j] for i=j)
                elif i == j:
                    part_str = 'd\u03C1/dt['+str(i)+','+str(j)+']'+' = -\u03B3['+str(i)+','+str(j)+']\u03C1['+str(i)+','+str(j)+']'
                    for k in range(i+1,self.num):
                        part_str += ' + \u03B3['+str(i)+','+str(k)+']\u03C1['+str(k)+','+str(k)+']'
                    part_str += ' + iE/\u0127('
                    for k in range(i):
                        if self.mu[k,i] == 0:
                            pass
                        else:
                            part_str += '+\u03BC['+str(k)+','+str(i)+'](\u03C1['+str(k)+','+str(i)+']-\u03C1['+str(i)+','+str(k)+'])'
                    for k in range(i+1,self.num):
                        if self.mu[i,k] == 0:
                            pass
                        else:
                            part_str += '+\u03BC['+str(i)+','+str(k)+'](\u03C1['+str(k)+','+str(i)+']-\u03C1['+str(i)+','+str(k)+'])'
                    part_str += ')\n\n'
                    main_str += part_str
                #generate non diagonal elements (drho_dt[i,j] for i<j):
                elif i < j:
                    part_str = 'd\u03C1/dt['+str(i)+','+str(j)+']'+' = -\u0393bar['+str(i)+','+str(j)+']\u03C1['+str(i)+','+str(j)+'] + i\u03C9['+str(j)+','+str(i)+']\u03C1['+str(i)+','+str(j)+']'
                    part_str += ' + iE/\u0127('
                    if self.mu[i,j] == 0:
                        pass
                    else:
                        part_str += '\u03BC['+str(i)+','+str(j)+'](\u03C1['+str(j)+','+str(j)+']-\u03C1['+str(i)+','+str(i)+'])'
                    for k in range(i):
                        if k == j or self.mu[k,i] == 0:
                            pass
                        else:
                            part_str += '+\u03BC['+str(k)+','+str(i)+']\u03C1['+str(k)+','+str(j)+']'
                    for k in range(i+1,self.num):
                        if k == j or self.mu[i,k] == 0:
                            pass
                        else:
                            part_str += '+\u03BC['+str(i)+','+str(k)+']\u03C1['+str(k)+','+str(j)+']'
                    for k in range(j):
                        if k == i or self.mu[k,j] == 0:
                            pass
                        else:
                            part_str += '-\u03BC['+str(k)+','+str(j)+']\u03C1['+str(i)+','+str(k)+']'
                    for k in range(j+1,self.num):
                        if k == i or self.mu[j,k] == 0:
                            pass
                        else:
                            part_str += '-\u03BC['+str(j)+','+str(k)+']\u03C1['+str(i)+','+str(k)+']'
                    part_str += ')\n\n'
                    main_str += part_str
                #non diagonal elements (drho_dt[i,j] for i>j) are skipped:
                else:
                    pass
        main_str = main_str.replace('(+','(')
        #print rate equation
        if string:
            return main_str
        else:
            print(main_str)
            return


#########################
##### RWA extension #####
#########################
#The code below contains systemrwa class for rotating wave approximation


class systemrwa(system):
    def __init__(self, size, gamma, Gamma, mu, energy, rho_init):
        super().__init__(size, gamma, Gamma, mu, energy, rho_init)
        self._rwa = True

    def print_rate_eq(self):
        '''Print density matrix rate equation.'''
        #Start writing rate equation:
        main_str = ''
        for i in range(self.num):
            for j in range(self.num):
                #generate drho_dt[0,0]:
                if (i,j) == (0,0):
                    part_str = 'd\u03C1/dt[0,0] ='
                    for k in range(1,self.num):
                        part_str += ' -d\u03C1/dt['+str(k)+','+str(k)+']'
                    part_str += '\n\n'
                    main_str += part_str
                #generate diagonal elements (drho_dt[i,j] for i=j):
                elif i == j:
                    part_str = 'd\u03C1/dt['+str(i)+','+str(j)+']'+' = -\u03B3['+str(i)+','+str(j)+']\u03C1['+str(i)+','+str(j)+']'
                    for k in range(i+1,self.num):
                        part_str += ' + \u03B3['+str(i)+','+str(k)+']\u03C1['+str(k)+','+str(k)+']'
                    part_str += ' + i/(2\u0127)('
                    for k in range(i):
                        if self.mu[k,i] == 0:
                            pass
                        else:
                            part_str += '+\u03BC['+str(k)+','+str(i)+'](\u03C1['+str(k)+','+str(i)+']E-\u03C1['+str(i)+','+str(k)+']E*)'
                    for k in range(i+1,self.num):
                        if self.mu[k,i] == 0:
                            pass
                        else:
                            part_str += '+\u03BC['+str(i)+','+str(k)+'](\u03C1['+str(k)+','+str(i)+']E*-\u03C1['+str(i)+','+str(k)+']E)'
                    part_str += ')\n\n'
                    main_str += part_str
                #generate non diagonal elements of vibronic transitions (drho_dt[i,j] for i<j):
                elif self.vibronic_trans[i,j] == 1:
                    part_str = 'd\u03C1/dt['+str(i)+','+str(j)+']'+' = -\u0393bar['+str(i)+','+str(j)+']\u03C1['+str(i)+','+str(j)+'] + i(\u03C9['+str(j)+','+str(i)+']-\u03C9\u2080)\u03C1['+str(i)+','+str(j)+']'
                    part_str += ' + i/(2\u0127)E*(\u03BC['+str(i)+','+str(j)+'](\u03C1['+str(j)+','+str(j)+']-\u03C1['+str(i)+','+str(i)+'])'
                    for k in range(i):
                        if k == j or self.mu[k,i] == 0:
                            pass
                        else:
                            part_str += '+\u03BC['+str(k)+','+str(i)+']\u03C1['+str(k)+','+str(j)+']'
                    for k in range(i+1,self.num):
                        if k == j or self.mu[i,k] == 0:
                            pass
                        else:
                            part_str += '+\u03BC['+str(i)+','+str(k)+']\u03C1['+str(k)+','+str(j)+']'
                    for k in range(j):
                        if k == i or self.mu[k,j] == 0:
                            pass
                        else:
                            part_str += '-\u03BC['+str(k)+','+str(j)+']\u03C1['+str(i)+','+str(k)+']'
                    for k in range(j+1,self.num):
                        if k == i or self.mu[j,k] == 0:
                            pass
                        else:
                            part_str += '-\u03BC['+str(j)+','+str(k)+']\u03C1['+str(i)+','+str(k)+']'
                    part_str += ')\n\n'
                    main_str += part_str
                #generate non diagonal elements of vibrational transitions (drho_dt[i,j] for i<j):
                elif self.vibrational_trans[i,j] == 1:
                    part_str = 'd\u03C1/dt['+str(i)+','+str(j)+']'+' = -\u0393bar['+str(i)+','+str(j)+']\u03C1['+str(i)+','+str(j)+'] + i\u03C9['+str(j)+','+str(i)+']\u03C1['+str(i)+','+str(j)+']'
                    part_str += ' + i/(2\u0127)('
                    for k in range(i):
                        if k == j or self.mu[k,i] == 0:
                            pass
                        else:
                            part_str += '+\u03BC['+str(k)+','+str(i)+']\u03C1['+str(k)+','+str(j)+']E'
                    for k in range(i+1,self.num):
                        if k == j or self.mu[i,k] == 0:
                            pass
                        else:
                            part_str += '+\u03BC['+str(i)+','+str(k)+']\u03C1['+str(k)+','+str(j)+']E*'
                    for k in range(j):
                        if k == i or self.mu[k,j] == 0:
                            pass
                        else:
                            part_str += '-\u03BC['+str(k)+','+str(j)+']\u03C1['+str(i)+','+str(k)+']E*'
                    for k in range(j+1,self.num):
                        if k == i or self.mu[j,k] == 0:
                            pass
                        else:
                            part_str += '-\u03BC['+str(j)+','+str(k)+'*\u03C1['+str(i)+','+str(k)+']E'
                    part_str += ')\n\n'
                    main_str += part_str
                #non diagonal elements (drho_dt[i,j] for i>j) are skipped:
                else:
                    pass
        main_str = main_str.replace('(+','(')
        #print rate equation
        print(main_str)
        return
        

