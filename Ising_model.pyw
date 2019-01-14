import sys, re
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, QDate, Qt
from PyQt5.QtGui import QIcon, QPixmap, QFont, QImage

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

from Ising_class import IsingModel, MagnetizationTemperature

dir_principal = os.getcwd()
carpeta_data = dir_principal + '\\Ising_data'

if not os.path.exists(carpeta_data): os.mkdir(carpeta_data)

class Window(QMainWindow): 
	def __init__(self):
		QMainWindow.__init__(self)
		os.chdir(carpeta_data)
		uic.loadUi('IsingModel.ui', self)
		os.chdir(dir_principal)

		self.showMaximized()

		#Animation and simulations
		self.animation.clicked.connect(self.fer_anim)
		self.simulate.clicked.connect(self.fer_simu)
		self.simulate_2.clicked.connect(self.fer_simu_temperatures)

		#Save files
		self.save_files = Save_files()

		self.save_magnet.clicked.connect(self.obrir_save_files_magnet)
		self.save_config.clicked.connect(self.obrir_save_files_config)
		self.save_energy.clicked.connect(self.obrir_save_files_energy)
		self.save_energy.clicked.connect(self.obrir_save_files_autocorrel)
		self.save_cv_T.clicked.connect(self.obrir_save_files_Cv_T)
		self.save_chi_T.clicked.connect(self.obrir_save_files_susceptibility_T)
		self.save_magnet_T.clicked.connect(self.obrir_save_files_magnet_T)
		self.save_energy_T.clicked.connect(self.obrir_save_files_energy_T)
		self.save_alpha_reg.clicked.connect(self.obrir_save_files_alpha_reg)
		self.save_beta_reg.clicked.connect(self.obrir_save_files_beta_reg)
		self.save_gamma_reg.clicked.connect(self.obrir_save_files_gamma_reg)

		#Plots
		self.configurations.clicked.connect(self.veure_grafic_config)
		self.magnetization.clicked.connect(self.veure_grafic_magnet)
		self.energy.clicked.connect(self.veure_grafic_energy)
		self.autocorrel.clicked.connect(self.veure_grafic_autocorrel)

		#T evolution
		self.magnet_T.clicked.connect(self.veure_grafic_magnet_T)
		self.energy_T.clicked.connect(self.veure_grafic_energy_T)
		self.cv_T.clicked.connect(self.veure_grafic_cv_T)
		self.chi_T.clicked.connect(self.veure_grafic_chi_T)

		#Regressions
		self.alpha_reg.clicked.connect(self.veure_grafic_alpha)
		self.beta_reg.clicked.connect(self.veure_grafic_beta)
		self.gamma_reg.clicked.connect(self.veure_grafic_gamma)

		self.show_results.clicked.connect(self.results)
		self.show_exponents.clicked.connect(self.exponents)

	def obrir_save_files_magnet(self):
		self.save_files.saveFileDialog('Magnetization_plot')

	def obrir_save_files_config(self):
		self.save_files.saveFileDialog('Configuration_plots')

	def obrir_save_files_energy(self):
		self.save_files.saveFileDialog('Energy_plot')

	def obrir_save_files_autocorrel(self):
		self.save_files.saveFileDialog('Autocorrelation_function')

	def obrir_save_files_energy_T(self):
		self.save_files.saveFileDialog('Energy_vs_temperature')

	def obrir_save_files_magnet_T(self):
		self.save_files.saveFileDialog('Magnetization_vs_temperature')

	def obrir_save_files_Cv_T(self):
		self.save_files.saveFileDialog('Cv_vs_temperature')

	def obrir_save_files_susceptibility_T(self):
		self.save_files.saveFileDialog('Susceptibility_vs_temperature')

	def obrir_save_files_alpha_reg(self):
		self.save_files.saveFileDialog('alpha_regression')

	def obrir_save_files_beta_reg(self):
		self.save_files.saveFileDialog('beta_regression')

	def obrir_save_files_gamma_reg(self):
		self.save_files.saveFileDialog('gamma_regression')

	def veure_grafic_config(self):
		self.imageLabel.clear()
		filename = 'Configuration_plots.png'

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must first simulate the system!')

	def veure_grafic_magnet(self):
		self.imageLabel.clear()
		filename = 'Magnetization_plot.png'

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must first simulate the system!')

	def veure_grafic_energy(self):
		self.imageLabel.clear()
		filename = 'Energy_plot.png'

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must first simulate the system!')

	def veure_grafic_autocorrel(self):
		self.imageLabel.clear()
		filename = 'Autocorrelation_function.png'

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must first simulate the system!')

	def veure_grafic_magnet_T(self):
		self.imageLabel.clear()
		filename = 'Magnetization_vs_temperature.png'

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must first simulate the system!')

	def veure_grafic_energy_T(self):
		self.imageLabel.clear()
		filename = 'Energy_vs_temperature.png'

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must first simulate the system!')

	def veure_grafic_cv_T(self):
		self.imageLabel.clear()
		filename = 'Cv_vs_temperature.png'

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must first simulate the system!')

	def veure_grafic_chi_T(self):
		self.imageLabel.clear()
		filename = 'Susceptibility_vs_temperature.png'

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must first simulate the system!')

	def veure_grafic_alpha(self):
		self.imageLabel.clear()
		filename = 'alpha_regression.png'

		if os.path.exists(filename):
			image = QImage(filename)
			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must first simulate the system!')

	def veure_grafic_beta(self):
		self.imageLabel.clear()
		filename = 'beta_regression.png'

		if os.path.exists(filename):
			image = QImage(filename)
			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must first simulate the system!')

	def veure_grafic_gamma(self):
		self.imageLabel.clear()
		filename = 'gamma_regression.png'

		if os.path.exists(filename):
			image = QImage(filename)
			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must first simulate the system!')

	def fer_simu(self):
		N = self.rows.value() #length of the lattice
		M = self.columns.value() #width of the lattice

		N_dipols = N*M

		prob_up = self.probability.value() #Probability to init a spin up in %

		t = self.simu_time.value() #Simulation time
		eq_time = self.eq_time.value() #Equilibrium time

		T = self.temperature.value() #Temperature of the system in kelvin
		H = self.magnetic_intensity.value() #Magnetic field intensity (Usually set to 0)
		J = self.strength.value() #Strength of the neighbours interaction
		freq = self.frequency.value() #Calculate total energy for multiples of this number

		self.ising = IsingModel(N, M, prob_up, t, eq_time, T, H, J, freq)
		self.ising.experiment_simulation()

		QMessageBox.information(self, 'Information', 'Simulation finished!')

	def fer_anim(self):
		N = self.rows.value() #length of the lattice
		M = self.columns.value() #width of the lattice

		N_dipols = N*M

		prob_up = self.probability.value() #Probability to init a spin up in %

		t = self.simu_time.value() #Simulation time
		eq_time = self.eq_time.value() #Equilibrium time

		T = self.temperature.value() #Temperature of the system in kelvin
		H = self.magnetic_intensity.value() #Magnetic field intensity (Usually set to 0)
		J = self.strength.value() #Strength of the neighbours interaction
		freq = self.frequency.value()

		self.ising = IsingModel(N, M, prob_up, t, eq_time, T, H, J, freq)
		self.ising.experiment_animation()

	def fer_simu_temperatures(self):
		N = self.rows.value() #length of the lattice
		M = self.columns.value() #width of the lattice

		N_dipols = N*M

		prob_up = self.probability.value() #Probability to init a spin up in %

		t = self.simu_time.value() #Simulation time
		eq_time = self.eq_time.value() #Equilibrium time

		#T = self.temperature.value() #Temperature of the system in kelvin
		H = self.magnetic_intensity.value() #Magnetic field intensity (Usually set to 0)
		J = self.strength.value() #Strength of the neighbours interaction
		freq = self.frequency.value() #Calculate total energy for multiples of this number

		initial = self.initial_T.value()
		final = self.final_T.value()
		n = self.nT.value()

		if self.checkBox.isChecked():
			logical_value = True
		else:
			logical_value = False

		self.evolution_experiment = MagnetizationTemperature(N, M, prob_up, t, eq_time, H, J, initial, final, n, freq, logical_value)
		self.evolution_experiment.experiment()

		QMessageBox.information(self, 'Information', 'Simulation finished!')

	def results(self):
		os.chdir(carpeta_data)
		if os.path.exists('Ising_data.txt'):
			f = open('Ising_data.txt')
			file = f.readlines()
			f.close()

			lines = []
			for line in file:
				lines.append(line)

			if float(lines[0]) > 60:
				minutes = round(float(lines[0])/60, 2)
				self.time_spent.setText(str(minutes) + ' min')
			else:
				self.time_spent.setText(str(lines[0]) + 's')

			if len(lines) >= 2: 
				self.avg_magnet.setText(str(lines[1]) + 'Am^2/spin')
			if len(lines) >= 3:
				self.avg_energy.setText(str(lines[2]) + 'J')

	def exponents(self):
		if os.path.exists('exponents.txt'):
			f = open('exponents.txt')
			file = f.readlines()
			f.close()

			lines = []
			for line in file:
				lines.append(line)

			self.alpha_exp.setText(str(lines[0]))
			self.beta_exp.setText(str(lines[1]))
			self.gamma_exp.setText(str(lines[2]))

		else:
			QMessageBox.warning(self, 'Warning!', 'You must first simulate the system!')

	def closeEvent(self, event):
		os.chdir(carpeta_data)
		result = QMessageBox.question(self, 'Leaving...','Do you want to exit?', QMessageBox.Yes | QMessageBox.No)
		if result == QMessageBox.Yes:
			event.accept()
			if os.path.exists('Ising_data.txt'): os.remove('Ising_data.txt')
			if os.path.exists('exponents.txt'): os.remove('exponents.txt')
			if os.path.exists('Configuration_plots.png'): os.remove('Configuration_plots.png')
			if os.path.exists('Magnetization_plot.png'): os.remove('Magnetization_plot.png')
			if os.path.exists('Energy_plot.png'): os.remove('Energy_plot.png')
			if os.path.exists('Autocorrelation_function.png'): os.remove('Autocorrelation_function.png')
			if os.path.exists('Magnetization_vs_temperature.png'): os.remove('Magnetization_vs_temperature.png')
			if os.path.exists('Energy_vs_temperature.png'): os.remove('Energy_vs_temperature.png')
			if os.path.exists('Cv_vs_temperature.png'): os.remove('Cv_vs_temperature.png')
			if os.path.exists('Susceptibility_vs_temperature.png'): os.remove('Susceptibility_vs_temperature.png')
			if os.path.exists('alpha_regression.png'): os.remove('alpha_regression.png')
			if os.path.exists('beta_regression.png'): os.remove('beta_regression.png')
			if os.path.exists('gamma_regression.png'): os.remove('gamma_regression.png')
		else:event.ignore()

class Save_files(QFileDialog):
	def __init__(self):
		QFileDialog.__init__(self)

		self.title = 'Save files'
		self.left = 10
		self.top = 10
		self.width = 640
		self.height = 400 

		self.initUI()

	def initUI(self):
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

	def saveFileDialog(self, name):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog

		fileName, _ = QFileDialog.getSaveFileName(self, 'Save files') 

		if fileName:
			os.chdir(carpeta_data)
			if os.path.exists('%s.png' % name): copyfile('%s.png' % name, fileName + '.png')
			else: QMessageBox.warning(self, 'Warning!', 'The plot doesn\'t exist!') 

app = QApplication(sys.argv)
_window=Window()
_window.show()
app.exec_()