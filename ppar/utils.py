#  SPDX-License-Identifier: GPL-3.0+
#
# Copyright © 2025 T. Beck.
#
# This file is part of ppar.
#
# ppar is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ppar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ppar.  If not, see <http://www.gnu.org/licenses/>.

'''Auxiliary functions for ppar'''

import re

import numpy as np

from scipy.special import erfc
from scipy.constants import physical_constants
from scipy.stats import gaussian_kde,norm
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
	'''Gaussian for histogram fit.

	:param args:
		(area, mean, sigma)
	'''

	return args[0]/np.sqrt(2*np.pi)/args[2]*np.exp(-(x-args[1])**2/(2*args[2]**2))

def skewed_gaussian(x,*args):
	'''Skewed Gaussian for histogram fit.

	:param args:
		(area, scale, x0, skewness)
	'''

	t = (x-args[2])/args[1]

	return (2*args[0]/args[1])*norm.pdf(t)*norm.cdf(args[3]*t)

def exp(x,*args):
	'''Exponential function for histogram fit.'''

	return args[0]*np.exp(args[1]*x)
	#return args[0]*np.exp(args[1]*(x-args[2]))

def exp_gaussian(x,*args):

	return exp(x,*args[:2]) + gaussian(x,*args[2:])

def left(x,gauss,*args):
	'''Tail function for histogram fit.'''

	if gauss:
		return exp(x,*args[:2]) + gaussian(x,*args[2:])

	return exp(x,*args)

def right(x,*args):
	'''Peak function for histogram fit.'''

	return 0.5*args[0]*(erfc((x-args[2]-0.5*args[1])/(2*args[3]**2))-\
			    erfc((x-args[2]+0.5*args[1])/(2*args[3]**2)))

def two_erf(x,*args):

	return right(x,*args)

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

# ELEMENTS = ['Nn',
# 	    'H',
# 	    'He',
# 	    'Li',
# 	    'Be',
# 	    'B',
# 	    'C',
# 	    'N',
# 	    'O',
# 	    'F',
# 	    'Ne',
# 	    'Na',
# 	    'Mg',
# 	    'Al',
# 	    'Si',
# 	    'P',
# 	    'S',
# 	    'Cl',
# 	    'Ar',
# 	    'K',
# 	    'Ca',
# 	    'Sc',
# 	    'Ti',
# 	    'V',
# 	    'Cr',
# 	    'Mn',
# 	    'Fe',
# 	    'Co',
# 	    'Ni',
# 	    'Cu',
# 	    'Zn',
# 	    'Ga',
# 	    'Ge',
# 	    'As',
# 	    'Se',
# 	    'Br',
# 	    'Kr',
# 	    'Rb',
# 	    'Sr',
# 	    'Y',
# 	    'Zr',
# 	    'Nb',
# 	    'Mo',
# 	    'Tc',
# 	    'Ru',
# 	    'Rh',
# 	    'Pd',
# 	    'Ag',
# 	    'Cd',
# 	    'In',
# 	    'Sn',
# 	    'Sb',
# 	    'Te',
# 	    'I',
# 	    'Xe',
# 	    'Cs',
# 	    'Ba',
# 	    'La',
# 	    'Ce',
# 	    'Pr',
# 	    'Nd',
# 	    'Pm',
# 	    'Sm',
# 	    'Eu',
# 	    'Gd',
# 	    'Tb',
# 	    'Dy',
# 	    'Ho',
# 	    'Er',
# 	    'Tm',
# 	    'Yb',
# 	    'Lu',
# 	    'Hf',
# 	    'Ta',
# 	    'W',
# 	    'Re',
# 	    'Os',
# 	    'Ir',
# 	    'Pt',
# 	    'Au',
# 	    'Hg',
# 	    'Tl',
# 	    'Pb',
# 	    'Bi',
# 	    'Po',
# 	    'At',
# 	    'Rn',
# 	    'Fr',
# 	    'Ra',
# 	    'Ac',
# 	    'Th',
# 	    'Pa',
# 	    'U',
# 	    'Np',
# 	    'Pu',
# 	    'Am',
# 	    'Cm',
# 	    'Bk',
# 	    'Cf',
# 	    'Es',
# 	    'Fm',
# 	    'Md',
# 	    'No',
# 	    'Lr',
# 	    'Rf',
# 	    'Db',
# 	    'Sg',
# 	    'Bh',
# 	    'Hs',
# 	    'Mt',
# 	    'Ds',
# 	    'Rg',
# 	    'Cn',
# 	    'Nh',
# 	    'Fl',
# 	    'Mc',
# 	    'Lv',
# 	    'Ts',
# 	    'Og',
# 	   ]

# def name_nucl_new(A,Z):

# 	_A,_Z	= int(A),int(Z)
# 	element = ELEMENTS[_Z]

# 	return '%i%s'% (_A,element),r'$^{%i}$%s'% (_A,element)

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

def calc_mode(data):
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

def calc_sc(data):
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
