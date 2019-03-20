########################################### 
"""
Project: Quantum Mechanics Non-linearities
Description: This code evolves states under the linear and non-linear Hamiltonians
Functions: 
"""
########################################### 

# Import dependencies 
from qutip import *
import yaml
import matplotlib.pyplot as plt
import os
import datetime
import time
import numpy as np
from scipy.special import factorial

class Solver(object):
	"""
	Description: Class that includes the simulation run and all the functions. 
	Input: Config file name
	Output: Data and plots
	"""
	def __init__(self, config):
		""" 
		Description: Class function, reads parameters from config file 
		Input: Config file
		Output: None 
		"""

		# Read parameters from file. If it fails, print fail statement
		self.args = {}
		if type(config) == str:
			with open(config) as cfile:
				self.args.update(yaml.load(cfile))
		elif type(config) == dict:
			self.args.update(config)
		else:
			print("Failed to load config arguments")

		# Assign parameters to class variables
		self.time = float(self.args['time']) # Final evolution time 
		self.N = int(self.args['N']) # Hilbert space size
		self.omegaC = float(self.args['omegaC']) # optical frequency (rescaled)
		self.C1bar = float(self.args['C1bar']) # single-photon coupling 
		self.muc = float(self.args['muc']) # optical coherent state parameter
		self.mum = float(self.args['mum']) # Mechanical coherent state parameter
		self.folder = str(self.args['folder']) # Folder where things are saved 

		# For time-dependent systems
		self.epsilon = float(self.args['epsilon']) # amplitude of squeezing
		self.Omega0 = float(self.args['Omega0']) # amplitude of squeezing

		# Noisy dynamics
		self.gammac = float(self.args['gammac']) # Optical noise
		self.gammam = float(self.args['gammam']) # Mechanical noise

	def coherent_coherent(self):
		"""
		Description: Generates an initial coherent state
		Input: None (thus far)
		Output: Separable initial state
		"""
		state = tensor(coherent(self.N, self.muc), coherent(self.N, self.mum))
		return state


	def coherent_thermal(self):
		"""
		Description: Generates a single thermal state
		Input: None
		Output: A thermal state
		"""
		states = []
		for i in range(0,self.N):
			states.append(tanh(self.rT)**(2*i)/cosh(rT)**2*fock_dm(self.N,i))
		thermal = sum(states)
		print(thermal.overlap(thermal))

		return tensor(coherent(self.N, self.mu), thermal)


	def run_time_independent(self, state):
		"""
		Description: Evolves the state
		Input: None
		Output: Array of [times, linear states, non-linear states]
		"""

		# Define operators
		a = tensor(destroy(self.N),qeye(self.N))
		b = tensor(qeye(self.N),destroy(self.N))
		d_ops = []
		
		# Build the non-linear Hamiltonians
		HNL = b.dag()*b - self.C1bar*a.dag()*a*(b + b.dag())

		# Define collapse operators
		Loptics = np.sqrt(self.gammac)*a
		Lmechanics = np.sqrt(self.gammam)*b

		# Define a list of all the expectation values we want
		if (self.gammac == 0 and self.gammam == 0):
			c_ops = []
		else:
			c_ops = [Loptics, Lmechanics]

		# Define array of times to feed the solver
		times = np.linspace(0.0, self.time, 100.0)

		# Call Master Equation solver. Give it Hamiltonian, state, times and decoherence ops. 
		# Also give is the list of expectation values to compute
		resultsNL = mesolve(HNL, state, times, c_ops = c_ops, e_ops = [], args = [], progress_bar = True)

		# Extract expectation values
		states = resultsNL.states

		return [times, states]

	def run_time_dependent(self, state):
		"""
		Description: Evolves the state
		Input: None
		Output: Array of [times, linear states, non-linear states]
		"""

		# Define operators
		a = tensor(destroy(self.N),qeye(self.N))
		b = tensor(qeye(self.N),destroy(self.N))
		d_ops = []
		
		# Define free Hamiltonian
		H0 = b.dag()*b 

		# Define the time-dependent Hamiltonian
		H1 = a.dag()*a*(b.dag() + b) 

		C1bar = self.C1bar
		epsilon = self.epsilon
		Omega0 = self.Omega0

		# Define a function for the coefficient:
		def H1_coeff(t, args):
			return - self.C1bar*(1. + epsilon*np.sin(Omega0*t))

		# Define the full Hamiltonian
		H = [H0, [H1, H1_coeff]]

		# Change the options so that we have a smaller stepsize 
		opts = Options()
		opts.order = 20
		opts.nsteps = 2500


		# Define collapse operators
		Loptics = np.sqrt(self.gammac)*a
		Lmechanics = np.sqrt(self.gammam)*b

		# Define a list of all the expectation values we want
		if (self.gammac == 0 and self.gammam == 0):
			c_ops = []
		else:
			c_ops = [Loptics, Lmechanics]

		# Define array of times to feed the solver
		times = np.linspace(0.0, self.time, 100.0)

		# Call Master Equation solver. Give it Hamiltonian, state, times and decoherence ops. 
		# Also give is the list of expectation values to compute
		results = mesolve(H, state, times, c_ops = c_ops, e_ops = [], args = [], progress_bar = True, options=opts)

		# Extract expectation values
		states = results.states

		return [times, states]


	def construct_sigmas(self, states):
		"""
		Description: Returns an array of CMs built from the second moments of the full states
		Input: Array of evalues
		Output: Array of CMs
		"""
		# Define operators
		a = tensor(destroy(self.N),qeye(self.N))
		b = tensor(qeye(self.N),destroy(self.N))

		# Optics quadratic exp values
		ada_exp = []
		aad_exp = []
		ad2_exp = []
		a2_exp = []

		# Mechanics quadratic exp values
		bdb_exp = []
		bbd_exp = []
		bd2_exp = []
		b2_exp = []

		# Mixed expectation values
		adb_exp = []
		abd_exp = []
		ab_exp = []
		adbd_exp = []

		# Single expectation values
		a_exp = []
		ad_exp = []
		b_exp = []
		bd_exp = []

		# Calculate exp values for each state
		for state in states:
			# Optics
			a_exp.append(expect(a, state))
			ada_exp.append(expect(a.dag()*a, state))
			
			# Mechanics		
			b_exp.append(expect(b, state))
			bdb_exp.append(expect(b.dag()*b, state))			

			# Mixed oeprators 
			abd_exp.append(expect(a*b.dag(), state))
			ab_exp.append(expect(a*b, state))

			# Single operators 
			a2_exp.append(expect(a*a, state))
			b2_exp.append(expect(b*b, state))

		# Define array of CMs
		sigmas = []

		# Build the CMs
		# Get an array of a CM at every different time
		for i in range(0,len(ada_exp)):
			# Optics 
			sigma11 = 1. + 2.*ada_exp[i] - 2.*a_exp[i]*np.conjugate(a_exp[i])
			sigma31 = 2.*a2_exp[i] - 2.*a_exp[i]*a_exp[i]

			# Mechanics
			sigma22 = 1. + 2.*bdb_exp[i]  - 2.*b_exp[i]*np.conjugate(b_exp[i])
			sigma42 = 2.*b2_exp[i] - 2.*b_exp[i]*b_exp[i]

			# Mixed sector 
			sigma21 = 2.*abd_exp[i] - 2.*a_exp[i]*np.conjugate(b_exp[i])
			sigma41 = 2.*ab_exp[i] - 2.*a_exp[i]*b_exp[i]

			sigma = np.matrix([[sigma11, np.conjugate(sigma21), np.conjugate(sigma31), np.conjugate(sigma41)], [sigma21, sigma22, np.conjugate(sigma41), np.conjugate(sigma42)], [sigma31, sigma41, sigma11, sigma21], [sigma41, sigma42, np.conjugate(sigma21), sigma22]])
			sigmas.append(sigma)

		# Return vector of CMs
		return sigmas

	def entropy(self,states):
		"""
		Description: Calculates entropies of states
		Input: Array of subsystem states
		Output: Array of linear entropies
		"""
		return [entropy_linear(state) for state in states]

	def optics_quads(self,states):
		""" 
		Description: Calculate quadratures
		Input: Array of states
		Output: Two arrays of x_exp and p_exp
		"""
		a = tensor(destroy(self.N), qeye(self.N))
		x = (a + a.dag())/np.sqrt(2.)
		p = 1.j * (a.dag() - a)/np.sqrt(2.)
		x_exp = []
		p_exp = []
		for state in states:
			x_exp.append(expect(x, state))
			p_exp.append(expect(p, state))
		return [x_exp, p_exp]

	def mechanics_quads(self,states):
		""" 
		Description: Calculate quadratures
		Input: Array of states
		Output: Two arrays of x_exp and p_exp
		"""
		b = tensor(qeye(self.N),destroy(self.N))
		x = (b + b.dag())/np.sqrt(2.)
		p = 1.j * (b.dag() - b)/np.sqrt(2.)
		x_exp = []
		p_exp = []
		for state in states:
			x_exp.append(expect(x, state))
			p_exp.append(expect(p, state))
		return [x_exp, p_exp]


	def CM_entropy(self, sigmas):
		"""
		Description: Returns the entropy of a covariance matrix
		Input: An array of CMs
		Output: An array of entropies
		"""
		eigenvalues = []
		S = []

		for sigma in sigmas:
			eigenvalues.append(np.linalg.eig(sigma))

		# Then compute the binary entropy for the smallest and largest eigenvalues 
		for eigenvalue in eigenvalues:
			vp = np.amax(eigenvalue[0])
			vm = np.amin(eigenvalue[0])
			s = (vp + 1.)/2. * np.log((vp + 1.)/2.) - (vp - 1.)/2. * np.log((vp-1.)/2.) + (vm + 1.)/2.*np.log((vm + 1.)/2.) - (vm - 1.)/2. * np.log((vm - 1.)/2.)
			S.append(s)

		return S

	def compute_nonGaussianity_time_independent(self):
		"""
		Description: Computes the non-Gaussianity of a quantum state
		Input: config.yaml
		Output: Three files in time-stamped folder: 
				Svalues = Values of non-Gaussianity
				times = Times used for computation 
				config = Current copy of config file
		"""

		# Define the initial state
		psi0 = self.coherent_coherent()
		print()

		# Obtain the states 
		[times, states] = self.run_time_independent(psi0)

		# Obtain the covariance matrices
		sigmas = self.construct_sigmas(states)

		# Obtain the entropies
		S = self.CM_entropy(sigmas)
		print(np.amax(S))
		
		# Create folder destination, check if exists if not create
		st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H.%M.%S')
		foldername = "data/" + "time_independent/" + "phonon_decoherence/" +  st
		if not os.path.exists(foldername):
			os.makedirs(foldername)
		
		with open(foldername + "/Svalues", 'w') as f: # Write the values to file
			for item in S:
				f.write("%s," % item)

		with open(foldername + "/times", 'w') as f: # Write the times to file, mostly because Mathematica generates different times to the Tsble function
			for item in times:
				f.write("%s," % item)

		# Add the current config file into the simulation folder for later identification
		with open(foldername + "/config", 'w') as outfile:
			outfile.write(yaml.dump(self.args, default_flow_style = False))
		
		# Plot the results	
		plt.figure(figsize=(10,7.5))
		plt.plot(times, S)
		plt.show()

	def compute_nonGaussianity_time_dependent(self):
		"""
		Description: Computes the non-Gaussianity of a quantum state for time-dependent dynamics
		Input: config.yaml
		Output: Three files in time-stamped folder: 
				Svalues = Values of non-Gaussianity
				times = Times used for computation 
				config = Current copy of config file
		"""

		# Define the initial state
		psi0 = self.coherent_coherent()
		print()

		# Obtain the states 
		[times, states] = self.run_time_dependent(psi0)

		# Obtain the covariance matrices
		sigmas = self.construct_sigmas(states)

		# Obtain the entropies
		S = self.CM_entropy(sigmas)
		print(np.amax(S))
		
		# Create folder destination, check if exists if not create
		st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H.%M.%S')
		foldername = "data/" + "time_dependent/" + "phonon_decoherence/" + st
		if not os.path.exists(foldername):
			os.makedirs(foldername)
		
		with open(foldername + "/Svalues", 'w') as f: # Write the values to file
			for item in S:
				f.write("%s," % item)

		# Add the current config file into the simulation folder for later identification
		with open(foldername + "/config", 'w') as outfile:
			outfile.write(yaml.dump(self.args, default_flow_style = False))

		with open(foldername + "/times", 'w') as f: # Write the times to file, mostly because Mathematica generates different times to the Tsble function
			for item in times:
				f.write("%s," % item)

		plt.figure(figsize=(10,7.5))
		plt.plot(times, S)
		plt.show()


#####################################
# Coefficient functions for plain optomechanics
# Defined in the paper
#####################################
	def FNa2(self, t):
		return - (self.C1bar**2)*(2.*t - np.sin(2.*t))
	def FNaBp(self, t):
		return - self.C1bar*np.sin(t)
	def FNaBm(self, t):
		return - self.C1bar*(np.cos(t)- 1.)

	def compute_a(self, states):
		a = tensor(destroy(self.N), qeye(self.N))
		a_exp = [expect(a, state) for state in states]

		return a_exp

	def compute_b(self, states):
		b = tensor(qeye(self.N),destroy(self.N))
		b_exp = [expect(b, state) for state in states]

		return b_exp

	def compute_ada(self, states):
		a = tensor(destroy(self.N), qeye(self.N))
		ada_exp = [expect(a.dag()*a, state) for state in states]

		return ada_exp

	def compute_bdb(self, states):
		b = tensor(qeye(self.N),destroy(self.N))
		bdb_exp = [expect(b.dag()*b, state) for state in states]

		return bdb_exp

	def compute_a2(self, states):
		a = tensor(destroy(self.N), qeye(self.N))
		a2_exp = [expect(a*a, state) for state in states]

		return a2_exp

	def compute_b2(self, states):
		b = tensor(qeye(self.N),destroy(self.N))
		b2_exp = [expect(b*b, state) for state in states]

		return b2_exp

	def compute_ab(self, states):
		a = tensor(destroy(self.N), qeye(self.N))
		b = tensor(qeye(self.N),destroy(self.N))
		ab_exp = [expect(a*b, state) for state in states]

		return ab_exp	

	def compute_adb(self, states):
		a = tensor(destroy(self.N), qeye(self.N))
		b = tensor(qeye(self.N),destroy(self.N))
		adb_exp = [expect(a.dag()*b, state) for state in states]

		return adb_exp	

	def compute_EBpBm(self, states):
		a = tensor(destroy(self.N), qeye(self.N))
		b = tensor(qeye(self.N),destroy(self.N))
		EBpBm_exp = [expect(a.dag()*b, state) for state in states]

if __name__ == "__main__":
	system = Solver('Noisy_config.yaml')
	print("Run time-independent system? (y/n)")
	if input() == "y":
		system.compute_nonGaussianity_time_independent()
	print("Run time-dependent system? (y/n)")
	if input() == "y":
		system.compute_nonGaussianity_time_dependent()
	