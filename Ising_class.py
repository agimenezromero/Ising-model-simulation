import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
import matplotlib.colors
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.patches as mpatches
from scipy import stats

import time
import os
from shutil import copyfile

#Define map colorsy  
levels = [-1, 0, 1]
colors = ['b', 'r']
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)

current_milli_time = lambda: int(round(time.time() * 1000))

dir_principal = os.getcwd()
carpeta_data = dir_principal + '\\Ising_data'

if not os.path.exists(carpeta_data): os.mkdir(carpeta_data)

class IsingModel(object):
	def __init__(self, N, M, prob_up, t, eq_time, T, H, J, frequency):
		self.N = N
		self.M = M

		self.prob_up = prob_up
		self.t = t
		self.eq_time = eq_time

		self.T = T
		self.H = H
		self.J = J
		self.frequency = frequency

		self.N_dipols = self.N * self.M
		self.lattice = np.zeros((N, M))

		k_B = 1 #1.380648e-23 #Boltzmann constant (J/K)
		self.beta = 1/(k_B*self.T)

	def init_random_lattice(self):
		for i in range(self.N):
			for j in range(self.M):
				value = random.randrange(0, 100)

				if value <= self.prob_up:
					self.lattice[i][j] = 1
				else:
					self.lattice[i][j] = -1

	def set_lattice(self, config):
		self.lattice = config

	def energy(self, i, j, H, J):

		Sn = self.lattice[(i - 1) % self.N, j] + self.lattice[(i + 1) % self.N, j] + self.lattice[i, (j - 1) % self.M] + self.lattice[i, (j + 1) % self.M]

		return (self.H - self.J * Sn) * self.lattice[i][j]

	def total_energy(self):
		E = 0
		for i in range(self.N):
			for j in range(self.M):
				E += self.energy(i, j, self.H, self.J)

		return E/(4 * self.N * self.M)

	def magnetization(self):
		return np.sum(self.lattice)/(self.N * self.M)

	def autocorr(self, x):
	    n = len(x)
	    variance = np.var(x)
	    x = x-np.mean(x)
	    r = np.correlate(x, x, mode = 'full')[-n:]
	    #assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
	    result = r /(variance*(np.arange(n, 0, -1)))
	    return result

	def metropolis(self): #Simulate Ising model using metropolis alorithm
		
		i = random.randrange(0,self.N) #Spins are equally probable to be choosen
		j = random.randrange(0, self.M)

		E_0 = self.energy(i, j, self.H, self.J)

		self.lattice[i][j] *= -1

		E_f = self.energy(i, j, self.H, self.J)

		delta_E = E_f-E_0

		prob = np.exp(-self.beta*delta_E)

		if delta_E < 0 or np.random.random() < prob: # Just keep the flipped value if the energy decreases, else keep only with probability e^(-\beta(delta_E))
			pass #accept the flip

		else:
			self.lattice[i][j] *= -1	

		return self.lattice		

	def animation(self):
		configurations = []
		
		def update(t, lattice, lines):

			lattice = self.metropolis()
			configurations.append(lattice)

			lines.set_data(lattice)

			return lines,

		fig, ax = plt.subplots()

		plt.title('Ising model 2D')

		lines = ax.matshow(self.lattice, cmap = cmap)

		anim = FuncAnimation(fig, update, fargs=(self.lattice, lines), frames=self.t,
		                    blit=True, interval=0.1, repeat=False)

		plt.show()

		return configurations

	def simulation(self):
		configurations = []
		magnetization_time = []
		energy_time = []
		MC_configs = []
		magnetization_MC = []

		for k in range(self.t):
			lattice = self.metropolis()
			configurations.append(lattice)
			magnetization_time.append(self.magnetization())
			
			if k % self.frequency == 0 :
				energy_time.append(self.total_energy())

			if k >= self.eq_time and k % (self.N * self.M) == 0:
				MC_configs.append(lattice)
				magnetization_MC.append(self.magnetization())


		return configurations, magnetization_time, energy_time, MC_configs, magnetization_MC

	def simulation_T(self):
		configurations = []
		magnetization_time = []
		energy_time = []

		for k in range(self.t):
			lattice = self.metropolis()
			configurations.append(lattice)

			if k >= self.eq_time and k % (self.N * self.M) == 0: #Monte carlo sweep after N*M iterations (aproximation)
				magnetization_time.append(self.magnetization())
				energy_time.append(self.total_energy()) 

		return configurations, magnetization_time, energy_time

	def experiment_animation(self):
 
		self.init_random_lattice()

		start = current_milli_time()
		config = self.animation()
		time_used = (current_milli_time() - start) / 1000

		#Configuration plots
		plt.subplot(121)
		plt.imshow(config[0], cmap=cmap)
		plt.title('Initial situation')

		plt.subplot(122)
		im = plt.imshow(config[self.t-1], cmap=cmap)
		plt.title('Final situation')

		plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
		cax = plt.axes([0.85, 0.1, 0.03, 0.8])
		plt.colorbar(cax=cax)

		os.chdir(carpeta_data)
		plt.savefig('Configuration_plots.png')
		plt.gcf().clear()

	def experiment_simulation(self):

		start = current_milli_time()

		self.init_random_lattice()

		plt.subplot(121)
		plt.imshow(self.lattice, cmap = cmap)
		plt.title('Initial situation')
		plt.xticks([])
		plt.yticks([])

		config, magnetization_time, energy_time, MC_configs, magnetization_MC = self.simulation()

		########################## PLOTS ###########################

		os.chdir(carpeta_data)

		#Configuration plots
		plt.subplot(122)
		im = plt.imshow(config[self.t-1], cmap=cmap)
		plt.title('Final situation')

		plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
		plt.xticks([])
		plt.yticks([])

		red_patch = mpatches.Patch(color='red', label='Spin up')
		blue_patch = mpatches.Patch(color='blue', label='Spin down')

		plt.suptitle('Spin system at T=%.2f' % self.T)
		plt.legend(bbox_to_anchor=(1.5, 1.5), handles=[red_patch, blue_patch], loc='upper right')


		plt.savefig('Configuration_plots.png')
		plt.gcf().clear()

		#Mitjana magnetització
		mitja_M = 0
		mitja_M_squared = 0

		for i in range(self.eq_time, len(magnetization_time)):
			mitja_M += magnetization_time[i]
			mitja_M_squared += magnetization_time[i]**2

		mitja_M = mitja_M/(len(magnetization_time)- self.eq_time)
		mitja_M_squared = mitja_M_squared/(len(magnetization_time)-self.eq_time)

		#Magnetization plots
		x = np.linspace(0, len(magnetization_time), len(magnetization_time))
		med = np.linspace(mitja_M, mitja_M, len(magnetization_time))

		plt.plot(x, magnetization_time, '-', label = 'Magnetization')

		if self.eq_time != 0:
			plt.plot(x, med, label = 'Average magnetization')

		plt.title('Evolution of magnetization in time at T=%.2f' % self.T)
		plt.xlabel('Time (s)')
		plt.ylabel('Magnetization A/m')
		plt.legend()

		plt.savefig('Magnetization_plot.png')
		plt.gcf().clear()

		#Mitjana energia
		mitja_E = 0

		for i in range(int(self.eq_time/self.frequency), len(energy_time)):
			mitja_E += energy_time[i]

		mitja_E = mitja_E/(len(energy_time)-self.eq_time/self.frequency)

		#Energy plots
		x = np.linspace(0, len(energy_time), len(energy_time))
		med = np.linspace(mitja_E, mitja_E, len(energy_time))

		plt.plot(x, energy_time, '-', label = 'Energy')

		if self.eq_time != 0:
			plt.plot(x, med, label = 'Average energy in equilibrium')

		plt.title('Evolution of energy in time at T=%.2f' % self.T)
		plt.xlabel('Time (s)')
		plt.ylabel('Energy J')
		plt.legend()

		plt.savefig('Energy_plot.png')
		plt.gcf().clear()
	
		#Numpy autocorrelation
		autocorrelation_np = self.autocorr(magnetization_MC)

		plt.plot(np.linspace(0, len(magnetization_MC), len(magnetization_MC)), autocorrelation_np)
		plt.title('Autocorrelation function of m at T=%.2f' % self.T)
		plt.xlabel('MC steps')
		plt.ylabel('Correlation')
		plt.savefig('Autocorrelation_function.png')
		plt.gcf().clear()

		#Dades al fitxer
		time_used = (current_milli_time() - start) / 1000

		f = open('Ising_data.txt', 'w')

		f.write(str(round(time_used, 2)) + '\n')
		f.write(str(round(mitja_M, 4)) + '\n')
		f.write(str(round(mitja_E, 4)) + '\n')

		f.close()

class MagnetizationTemperature(object):
	def __init__(self, N, M, prob_up, t, eq_time, H, J,init_T, final_T, n, freq, logical_value):
		self.N = N
		self.M = M
		self.prob_up = prob_up
		self.t = t
		self.eq_time = eq_time
		self.H = H
		self.J = J
		self.frequency = freq

		self.init_T = init_T
		self.final_T = final_T
		self.n = n

		self.Ts = list(np.linspace(init_T, final_T, n))

		if logical_value == True:
			self.Ts.append(2.27) #Append critical temperature
			self.Ts.sort()

		self.Ts = np.array(self.Ts)

	def experiment(self):

		magnetization = []
		energy = []
		energy_squared = []
		magnetization_squared = []
		configs = []

		start = current_milli_time()
		counter = 0

		for T in self.Ts:
			ising = IsingModel(self.N, self.M, self.prob_up, self.t, self.eq_time, T, self.H, self.J, self.frequency)

			if counter == 0:
				ising.init_random_lattice()
			else:
				ising.set_lattice(configs[counter-1][self.t-1]) #Set the last configuration of the previus T_simulation to assure
																#convergence to equilibrium (Patriah)

			config, magnetization_time, energy_time = ising.simulation_T()

			configs.append(config)

			#Mitjana magnetització per cada T
			mitja_exp = np.sum(magnetization_time)/len(magnetization_time)
			magnetization.append(abs(mitja_exp))

			mitja_M_squared = np.sum(np.array(magnetization_time)**2)/len(magnetization_time)
			magnetization_squared.append(mitja_M_squared)

			#Mitjana energia per cada T
			mitja_E = np.sum(energy_time)/len(energy_time)
			energy.append(mitja_E)

			mitja_E_squared = np.sum(np.array(energy_time)**2)/len(energy_time)
			energy_squared.append(mitja_E_squared)

			counter += 1

		#Magnetization plots
		plt.plot(self.Ts, magnetization, ls='-', marker='.', label=r'$m(T)$')
		plt.plot(np.linspace(2.27, 2.27, 100), np.linspace(0, 1, 100), color='r', label=r'$T_c$')		
		plt.title('Magnetization vs Temperature')
		plt.xlabel('T')
		plt.ylabel('m')
		plt.legend()
		
		os.chdir(carpeta_data)	
		plt.savefig('Magnetization_vs_temperature.png')
		plt.gcf().clear()

		#Susceptibility plots x=beta*(<m^2>-<m>^2)

		chi = []

		for i in range(len(energy)):
			beta = 1/self.Ts[i] #k_B=1
			chi.append(beta*(magnetization_squared[i] - magnetization[i]**2))

		plt.plot(self.Ts, chi, ls='-', marker='.', label=r'$\chi(T)$')
		plt.plot(np.linspace(2.27, 2.27, 100), np.linspace(0, max(chi), 100), color='r', label=r'$T_c$')
		plt.title('Susceptibility vs Temperature')
		plt.xlabel('T')
		plt.ylabel(r'$\chi$')
		plt.legend()


		plt.savefig('Susceptibility_vs_temperature.png')
		plt.gcf().clear()

		#Energy plots
		plt.plot(self.Ts, energy, ls='-', marker='.', label=r'$u(T)$')
		plt.plot(np.linspace(2.27, 2.27, 100), np.linspace(min(energy), max(energy), 100), color='r', label=r'$T_c$')		
		plt.title('Energy vs Temperature')
		plt.xlabel('T')
		plt.ylabel('E/N')
		plt.legend()
		
		os.chdir(carpeta_data)	
		plt.savefig('Energy_vs_temperature.png')
		plt.gcf().clear()

		#Specific heat plots Cv = beta^2*(<E^2>-<E>^2)
		Cv = []

		for i in range(len(energy)):
			beta = 1/self.Ts[i] #k_B=1
			Cv.append(beta**2*(energy_squared[i] - energy[i]**2))

		plt.plot(self.Ts, Cv, ls='-', marker='.', label=r'$c_V(T)$')
		plt.plot(np.linspace(2.27, 2.27, 100), np.linspace(0, max(Cv), 100), color='r', label=r'$T_c$')
		plt.title('Specific heat vs Temperature')
		plt.xlabel('T')
		plt.ylabel(r'$c_V$')
		plt.legend()

		plt.savefig('Cv_vs_temperature.png')
		plt.gcf().clear()

		#Critical exponent of magnetization m=|T-Tc|^\beta
		Tc = np.linspace(2.27, 2.27, len(self.Ts))

		log_m = np.log(magnetization)
		log_T = np.log(abs(self.Ts-Tc))

		slope_beta, intercept, r_value, p_value, std_err = stats.linregress(log_T, log_m)

		lines = slope_beta * log_T + intercept

		plt.plot(log_T, log_m, '.', label = 'Experimental data')
		plt.plot(log_T, lines, ls='-', label = 'Linear regression')
		plt.title(r'Regressió $\log{m} \ vs \ {\beta}\log{|T-T_c|}$')
		plt.legend()
		plt.savefig('beta_regression.png')
		plt.gcf().clear()

		#Critical exponent of heat capacity
		log_cv = np.log(Cv)

		slope_alpha, intercept, r_value, p_value, std_err = stats.linregress(log_T, log_cv)

		lines = slope_alpha * log_T + intercept

		plt.plot(log_T, log_cv, '.', label = 'Experimental data')
		plt.plot(log_T, lines, ls='-', label = 'Linear regression')
		plt.title(r'Regressió $\log{C_V} \ vs \ {\alpha}\log{|T-T_c|}$')
		plt.legend()
		plt.savefig('alpha_regression.png')
		plt.gcf().clear()

		#Critical exponent of susceptibility
		log_chi = np.log(chi)

		slope_gamma, intercept, r_value, p_value, std_err = stats.linregress(log_T, log_chi)

		lines = slope_gamma * log_T + intercept

		plt.plot(log_T, log_chi, '.', label = 'Experimental data')
		plt.plot(log_T, lines, ls='-', label = 'Linear regression')
		plt.title(r'Regressió $\log{\chi} \ vs \ {\gamma}\log{|T-T_c|}$')
		plt.legend()
		plt.savefig('gamma_regression.png')
		plt.gcf().clear()

		#Dades al fitxer
		time_used = (current_milli_time() - start) / 1000

		f = open('Ising_data.txt', 'w')
		f.write(str(round(time_used, 2)) + '\n')
		f.close()

		f2 = open('exponents.txt', 'w')
		f2.write(str(round(-slope_alpha, 4)) + '\n')
		f2.write(str(round(slope_beta, 4)) + '\n')
		f2.write(str(round(-slope_gamma, 4)) + '\n')
		f2.close()

		return self.Ts, magnetization, chi, energy, Cv

