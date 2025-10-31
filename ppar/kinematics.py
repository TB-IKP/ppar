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

'''Stopping in target and kinematics'''

import numpy as np

from scipy.constants import physical_constants
from scipy.interpolate import interp1d

from .messages import stopping_message,header_message

#---------------------------------------------------------------------------------------#
#		Conversions
#---------------------------------------------------------------------------------------#

def Brho_to_p(Brho,nucl):
	'''Convert Brho (Tm) to p (MeV/c).'''

	c 	= physical_constants['speed of light in vacuum'][0]

	p 	= Brho*nucl['q']*c*1e-6			#in MeV/c

	return p

def Brho_to_TKE(Brho,nucl):
	'''Convert Brho (Tm) to TKE (MeV).'''

	c 	= physical_constants['speed of light in vacuum'][0]
	e 	= physical_constants['elementary charge'][0]
	u 	= physical_constants['atomic mass constant'][0]

	p	= Brho_to_p(Brho,nucl)       		#in MeV
	E0 	= nucl['ion mass']*u/e*c**2*1e-6 	#in MeV
	Etot	= np.sqrt(E0**2+p**2)        		#in MeV

	return Etot-E0

def Brho_to_gamma(Brho,nucl):
	'''Convert Brho (Tm) to gamma.'''

	c 	= physical_constants['speed of light in vacuum'][0]
	e 	= physical_constants['elementary charge'][0]
	u 	= physical_constants['atomic mass constant'][0]

	gamma 	= np.sqrt(((nucl['q']*e*Brho)/(nucl['ion mass']*u*c))**2+1)

	return gamma

def Brho_to_beta(Brho,nucl):
	'''Convert Brho (Tm) to beta.'''

	gamma 	= Brho_to_gamma(Brho,nucl)
	beta 	= np.sqrt(1-1/(gamma**2))

	return beta

def TKE_to_p(TKE,nucl):
	'''Convert TKE (MeV) to p (MeV/c).'''

	c 	= physical_constants['speed of light in vacuum'][0]
	e 	= physical_constants['elementary charge'][0]
	u 	= physical_constants['atomic mass constant'][0]

	E0 	= nucl['ion mass']*u/e*c**2*1e-6 	#in MeV
	Etot 	= TKE+E0
	p	= np.sqrt(Etot**2-E0**2)

	return p

def p_to_Brho(p,nucl):
	'''Convert p (MeV/c) to Brho (Tm).'''

	c 	= physical_constants['speed of light in vacuum'][0]

	Brho 	= p/nucl['q']/c*1e6			#in Tm

	return Brho

#---------------------------------------------------------------------------------------#
#	NSCL:	Convert Brho to kinematic quantities
#---------------------------------------------------------------------------------------#

def kinematics_reaction(Brho,nucl):
	'''Convert Brho to relevant kinematic quantities.'''

	values 		= {}

	values['Brho'] 	= Brho
	values['nucl'] 	= nucl['name']

	values['p'] 	= Brho_to_p(Brho,nucl)
	values['TKE'] 	= Brho_to_TKE(Brho,nucl)
	values['gamma'] = Brho_to_gamma(Brho,nucl)
	values['beta'] 	= Brho_to_beta(Brho,nucl)

	return values

# def kinematic_reaction(Brho,nucl):
# 	'''Convert Brho to relevant kinematic quantities.'''

# 	positions 	= ['before','after']
# 	values 		= {pos:{} for pos in positions}

# 	for num,pos in enumerate(positions):

# 		values[pos]['Brho'] 	= Brho[pos]
# 		values[pos]['nucl'] 	= nucl[num]['name']

# 		values[pos]['p'] 	= Brho_to_p(Brho[pos],nucl[num])
# 		values[pos]['TKE'] 	= Brho_to_TKE(Brho[pos],nucl[num])
# 		values[pos]['gamma'] 	= Brho_to_gamma(Brho[pos],nucl[num])
# 		values[pos]['beta'] 	= Brho_to_beta(Brho[pos],nucl[num])

# 	return values

#---------------------------------------------------------------------------------------#
#	NSCL: 	Correct kinematic quantities from fit
#---------------------------------------------------------------------------------------#

def correct_kinematics(p,fit_res,nucl):
	'''Correction of kinematic quantities to center
	the momentum of the unreacted beam in the S800.'''

	#correct momentum for offset from fit
	p_corr 	= p + fit_res.x[-2]

	Brho 	= p_to_Brho(p_corr,nucl)
	values 	= kinematics_reaction(Brho,nucl)

	return values

#---------------------------------------------------------------------------------------#
#		Atima run
#---------------------------------------------------------------------------------------#

def run_atima(TKE,nucl,target):
	'''Run Atima via PyAtima and extract the range in the target material.'''

	import PyAtima as pa

	#convert TKE to MeV/u
	en 		= TKE/nucl['ion mass']

	atima		= pa.PyAtima(Zp=nucl['Z'],Ap=nucl['ion mass'],E=en,
					Zt=target['Z'],At=target['ion mass'],
					t=target['thickness'])

	#convert range to µm
	dist		= atima.results['range']/(target['density']*1e3*1e-4)
	std_dist	= atima.results['std_range']/(target['density']*1e3*1e-4)

	#energy out
	#en_out		= atima.results['E']
	#tke_out		= en_out*nucl['ion mass']

	#return [en,TKE,dist,std_dist,en_out,tke_out]
	return [en,TKE,dist,std_dist]

#---------------------------------------------------------------------------------------#
#	NSCL: 	Regula falsi
#---------------------------------------------------------------------------------------#

def start_guesses_fw(TKE_max,nucl,target):
	'''Get good start values for regula falsi - forward.'''

	#calculation at maximum TKE
	res_max = run_atima(TKE_max,nucl,target)

	#For TKE=0 also range=0 is obtained.
	#Use this and the fact that the function
	#range(TKE) is convex to estimate a lower point:
	#TKE_min = TKE_max*(r_max-d_target)/r_max
	TKE_min = TKE_max*(res_max[2]-target['d'])/res_max[2]
	res_min = run_atima(TKE_min,nucl,target)

	return res_min,res_max

def start_guesses_bw(TKE_min,nucl,target):
	'''Get good start values for regula falsi - backward.'''

	#calculation at minimum TKE (after target)
	res_min = run_atima(TKE_min,nucl,target)

	#For TKE=0 also range=0 is obtained.
	#Use this and the fact that the function
	#range(TKE) is convex to estimate an upper point:
	#TKE_max = TKE_min*(r_min+d_target)/r_min
	TKE_max = TKE_min*(res_min[2]+target['d'])/res_min[2]
	res_max = run_atima(TKE_max,nucl,target)

	return res_min,res_max

def regula_falsi_fw(res0,res1,threshold,nucl,target):
	'''Implementation of regula falsi - forward.'''

	iterations 	= 0

	#Initialization
	TKE0		= res0[1]

	#energy before reaction (TKE_max)
	TKE1		= res1[1]

	#range after target = range_before - target
	range_after 	= res1[2] - target['d']

	#results Atima runs
	data_points 	= np.vstack((res0,res1))

	res2_shifted 	= np.inf

	while np.abs(res2_shifted) > threshold:

		res0_shifted 	= res0[2] - range_after
		res1_shifted 	= res1[2] - range_after

		#TKE2 		= TKE0-(TKE1-TKE0)*res0_shifted/(res1_shifted-res0_shifted)
		TKE2		= (TKE0*res1_shifted-TKE1*res0_shifted)/(res1_shifted-res0_shifted)

		res2		= run_atima(TKE2,nucl,target)
		res2_shifted 	= res2[2] - range_after

		data_points 	= np.vstack((data_points,res2))

		if res0_shifted*res2_shifted < 0:

			TKE1 = TKE2

		else:
			TKE0 = TKE2

		iterations     += 1

	return data_points,iterations

def regula_falsi_bw(res0,res1,threshold,nucl,target):
	'''Implementation of regula falsi - backward.'''

	iterations 	= 0

	#energy after reaction (TKE_min)
	TKE0		= res0[1]

	#Initialization
	TKE1		= res1[1]

	#range before target = range_after + target
	range_before 	= res0[2] + target['d']

	#results Atima runs
	data_points 	= np.vstack((res0,res1))

	res2_shifted 	= np.inf

	while np.abs(res2_shifted) > threshold:

		res0_shifted 	= res0[2] - range_before
		res1_shifted 	= res1[2] - range_before

		TKE2		= (TKE0*res1_shifted-TKE1*res0_shifted)/(res1_shifted-res0_shifted)

		res2		= run_atima(TKE2,nucl,target)
		res2_shifted 	= res2[2] - range_before

		data_points 	= np.vstack((data_points,res2))

		if res0_shifted*res2_shifted < 0:

			TKE1 = TKE2

		else:
			TKE0 = TKE2

		iterations     += 1

	return data_points,iterations

#---------------------------------------------------------------------------------------#
#		Stopping in target
#---------------------------------------------------------------------------------------#

def stopping_target_fw(kinematics,threshold,beam,target,product,verbose):
	'''Correlations between TKE and range from Atima - forward.'''

	if verbose:

		header_message('Stopping')

	data_points 	= {}
	interpolation 	= {}

	#beam means that the beam nucleus is stopped
	#product means that the reaction product is stopped
	for key in ['beam','product']:

		nucl			= beam if key=='beam' else product

		#if the reaction happens at the beginning of the target
		#the removed nucleons carry away a fraction of the TKE
		corr_fac		= 1 if key=='beam' else product['ion mass']/beam['ion mass']

		#definition of the interval for regula falsi
		#TKE0,TKE1 		= 0,corr_fac*kinematics['before']['TKE']
		res0,res1 		= start_guesses_fw(corr_fac*kinematics['before']['TKE'],nucl,target)
		data_points[key],calls  = regula_falsi_fw(res0,res1,threshold,nucl,target)

		#If regula falsi finds a solution within just one or two iterations,
		#the function is almost linear and linear interpolation is justified.
		kind 			= 'cubic' if len(data_points[key])>=4 else 'linear'

		#interpolation of the resulting energy-range plots (forward and backward)
		interpolation[key] 	= {'TKE-range':interp1d(data_points[key][:,1],
								data_points[key][:,2],
								kind=kind),
					   'range-TKE':interp1d(data_points[key][:,2],
								data_points[key][:,1],
								kind=kind),
					  }

		if verbose:

			stopping_message('fw',data_points[key],
					calls,threshold,nucl,target)

	return {'data':		data_points,
		'interpol':	interpolation}

def stopping_target_bw(kinematics,threshold,beam,target,product,verbose):
	'''Correlations between TKE and range from Atima - backward.'''

	if verbose:

		header_message('Stopping')

	data_points 	= {}
	interpolation 	= {}

	#beam means that the beam nucleus is stopped
	#product means that the reaction product is stopped
	for key in ['beam','product']:

		nucl			= beam if key=='beam' else product

		#if the reaction happens at the beginning of the target
		#the removed nucleons carry away a fraction of the TKE
		corr_fac		= 1 if key=='product' else beam['ion mass']/product['ion mass']

		#definition of the interval for regula falsi
		#TKE0,TKE1 		= 0,corr_fac*kinematics['before']['TKE']
		res0,res1 		= start_guesses_bw(corr_fac*kinematics['after']['TKE'],nucl,target)
		data_points[key],calls  = regula_falsi_bw(res0,res1,threshold,nucl,target)

		#If regula falsi finds a solution within just one or two iterations,
		#the function is almost linear and linear interpolation is justified.
		kind 			= 'cubic' if len(data_points[key])>=4 else 'linear'

		#interpolation of the resulting energy-range plots (forward and backward)
		interpolation[key] 	= {'TKE-range':interp1d(data_points[key][:,1],
								data_points[key][:,2],
								kind=kind),
					   'range-TKE':interp1d(data_points[key][:,2],
								data_points[key][:,1],
								kind=kind),
					  }

		if verbose:

			stopping_message('bw',data_points[key],
					calls,threshold,nucl,target)

	return {'data':		data_points,
		'interpol':	interpolation}

def stopping_target(en_in,en_out,steps,beam,target,product):
	'''Correlations between TKE and range from Atima - RIBF data.'''

	energies 	= np.linspace(en_in,en_out,steps+1)
	data_points	= {}
	interpolation 	= {}

	#Reaction at end of target (stopping with beam)
	#or beginning (stopping with product)
	for key in ['beam','product']:

		nucl = beam if key=='beam' else product

		results = np.zeros((4,len(energies)))

		#loop over energies
		for num_en,en in enumerate(energies):

			#convert to TKE to use run_atima function
			TKE 			= en*nucl['ion mass']

			#run Atima
			results[:,num_en] 	= run_atima(TKE,nucl,target)

		data_points[key] = results.T

		#interpolation of the resulting energy-range plots
		interpolation[key] 	= {'TKE-range':interp1d(data_points[key][:,1],
							        data_points[key][:,2],
							        kind='cubic'),
					   'range-TKE':interp1d(data_points[key][:,2],
							        data_points[key][:,1],
							        kind='cubic'),
					  }

	return {'data':		data_points,
		'interpol':	interpolation}

#---------------------------------------------------------------------------------------#
#		Range in momentum due to reaction position
#---------------------------------------------------------------------------------------#

def momentum_range_fw(kinematics,stopping,beam,target,product):
	'''Momentum uncertainty due to reaction position - forward.'''

	#The effect of the kinetic energy lost through
	#nucleon removal has to be considered for a
	#reaction at the end of the target only.
	#The other case was already treated in stopping_target.

	p_after = np.zeros(2)

	#beam means that the beam nucleus is stopped
	#product means that the reaction product is stopped
	for num,key in enumerate(['beam','product']):

		#print(key)
		#print(stopping['data'][key])

		#nucl		= beam if key=='beam' else product
		#corr_fac	= 1 if key=='beam' else product['A']/beam['A']

		#TKE_before 	= kinematics['before']['TKE']*corr_fac
		#range_before 	= stopping['interpol'][key]['TKE-range'](TKE_before)

		#the range 'before the target' is always given by
		#the second result from the Atima calculations
		range_before 	= stopping['data'][key][1,2]
		range_after	= range_before - target['d']

		#print(range_before,range_after)

		corr_fac	= product['ion mass']/beam['ion mass'] if key=='beam' else 1

		TKE_after	= stopping['interpol'][key]['range-TKE'](range_after)*corr_fac
		p_after[num]	= TKE_to_p(TKE_after,product)

		#print(TKE_to_p(TKE_after,product))

	diff = np.abs(p_after[0]-p_after[1])

	return np.array([-0.5*diff,0.5*diff])#+kinematics['after']['p']
	#return np.sort(p_after)

def momentum_range_bw(kinematics,stopping,beam,target,product):
	'''Momentum uncertainty due to reaction position - backward.'''

	#The effect of the kinetic energy lost through
	#nucleon removal has to be considered for a
	#reaction at the beginning of the target only.
	#The other case was already treated in stopping_target.

	p_before = np.zeros(2)

	#beam means that the beam nucleus is stopped
	#product means that the reaction product is stopped
	for num,key in enumerate(['beam','product']):

		#the range 'after the target' is always given by
		#the first result from the Atima calculations
		range_after 	= stopping['data'][key][0,2]
		range_before	= range_after + target['d']

		#print(range_before,range_after)

		corr_fac	= beam['ion mass']/product['ion mass'] if key=='product' else 1

		TKE_before	= stopping['interpol'][key]['range-TKE'](range_before)*corr_fac
		p_before[num]	= TKE_to_p(TKE_before,beam)

		#print(TKE_to_p(TKE_after,product))

	diff = np.abs(p_before[0]-p_before[1])

	return np.array([-0.5*diff,0.5*diff])#+kinematics['after']['p']
	#return np.sort(p_after)

#def momentum_range(stopping,beam,target,product):
#	'''Momentum uncertainty due to reaction position - RIBF data.'''

#	p_after = np.zeros(2)

	#here comes a method to determine the
	#energy at the start of the target
	#
	#
	#
	#for now, use some dummy
#	p_after = np.array([-100,100])

#	return p_after