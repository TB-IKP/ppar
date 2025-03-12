'''Auxiliary functions for ppar'''

import re

import numpy as np

from scipy.special import erfc
from scipy.constants import physical_constants
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

#---------------------------------------------------------------------------------------#
#		Target thickness in µm
#---------------------------------------------------------------------------------------#

def mg_cm2_to_um(target):
	'''Convert the target thickness from mg/cm2 to µm.'''

	d_target 	= target['thickness']/target['density']*1e-2*1e3 #in µm
	target['d'] 	= d_target

	return target

#---------------------------------------------------------------------------------------#
#		Ion mass from atomic mass
#---------------------------------------------------------------------------------------#

def atomic_mass_to_ion_mass(nucl):
	'''Convert the atomic mass to ion mass by subtraction of electron masses.'''

	if 'q' not in nucl.keys():

		nucl['q'] = nucl['Z']

	nucl['ion mass'] = nucl['mass'] - \
				nucl['q']*physical_constants['electron mass in u'][0]

	return nucl

#---------------------------------------------------------------------------------------#
#		Fit functions
#---------------------------------------------------------------------------------------#

def piecewise(x,gauss,params):
	'''Piecewise function for the description of experimental momentum distributions.'''

	#add a shift in index
	shift = 3 if gauss else 0

	value = np.piecewise(x,
			[x<params[0],x>=params[0]],
			[lambda x: left(x,gauss,*params[1:3+shift]),
			 lambda x: right(x,*params[3+shift:])
			])

	return value

def gaussian(x,*args):
	'''Gaussian for histogram fit.'''

	return args[0]/np.sqrt(2*np.pi)/args[2]*np.exp(-(x-args[1])**2/(2*args[2]**2))

def exp(x,*args):
	'''Exponential function for histogram fit.'''

	return args[0]*np.exp(args[1]*x)
	#return args[0]*np.exp(args[1]*(x-args[2]))

def left(x,gauss,*args):
	'''Tail function for histogram fit.'''

	if gauss:
		return exp(x,*args[:2]) + gaussian(x,*args[2:])

	return exp(x,*args)

def right(x,*args):
	'''Peak function for histogram fit.'''

	return 0.5*args[0]*(erfc((x-args[2]-0.5*args[1])/(2*args[3]**2))-\
			    erfc((x-args[2]+0.5*args[1])/(2*args[3]**2)))

#---------------------------------------------------------------------------------------#
#	NSCL:	Extract Brho from Barney file
#---------------------------------------------------------------------------------------#

def extract_Brho(file_name,pos):
	'''Extract magnetic rigidity settings from A1900/ARIS Barney files.'''

	seg = 7  if pos == 'before' else 8
	bts = 33 if pos == 'before' else 34

	with open(file_name) as file:

		data 		= file.read()

		identifier 	= f'Seg[ \t]*{seg}:[ \t]*[+-]*[ \t]*[0-9]+.[0-9]+[ \t]*Tm'
		line 		= re.search(identifier,data)

		if not bool(line):

			identifier 	= f'BTS{bts}[ \t]*[+-]*[ \t]*[0-9]+.[0-9]+[ \t]*Tm'
			line 		= re.search(identifier,data)

		Brho = float(re.search('[0-9]+.[0-9]+',line.group(0)).group(0))

	return Brho

#---------------------------------------------------------------------------------------#
#		Nucleus name
#---------------------------------------------------------------------------------------#

def name_nucl(nucl):
	'''Decompose the string containing the name of the nucleus.'''

	pattern_A 	= '([0-9]{1,3})'
	pattern_Z	= '([A-Z]{1}[a-z]{0,1})'

	#Find the name
	name_A 		= re.search(pattern_A,nucl).group(0)
	name_Z 		= re.search(pattern_Z,nucl).group(0)

	#Find the gamma-ray energy and angle (if given)
	return r'$^{%i}$%s'% (int(name_A),name_Z)

#---------------------------------------------------------------------------------------#
#		Mode and shortest coverage interval
#---------------------------------------------------------------------------------------#

def mode(data):
	'''Mode of data.
	Can be replaced with az.plots.plot_utils.calculate_point_estimate('mode',data)
	but this does not use minimization.'''

	kde_plain 		= gaussian_kde(data)
	kde_func 		= lambda x: (-1)*kde_plain(x)

	min_dat,max_dat 	= np.min(data),np.max(data)
	minimum 		= minimize(kde_func,x0=np.median(data),
						method='TNC',tol=1e-5,
						bounds=((min_dat,max_dat),))

	return minimum.x[0]

def sc_interval(data):
	'''Shortest coverage interval of data.
	Replace by arviz.hdi(data,hdi_prob=0.68268949) in the future.'''

	#threshold 		= int(np.ceil(np.math.erf(1/np.sqrt(2))*len(data))) #=0.6826894921370859
	threshold 		= int(0.68268949*len(data))

	sorted_data 		= np.sort(data)
	all_intervals 		= np.array([[sorted_data[i],sorted_data[i+threshold]] \
						for i in range(0,len(sorted_data)-threshold)])

	min_index 		= np.argmin(np.diff(all_intervals))
	min_interval 		= [all_intervals[min_index,0],all_intervals[min_index,1]]

	return min_interval
