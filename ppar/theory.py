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

'''Treatment of theoretical momentum distributions'''

import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import minimize

from .spectrum import piecewise
from .utils import gaussian,right

#---------------------------------------------------------------------------------------#
#		theo class
#---------------------------------------------------------------------------------------#

class theo():

	def __init__(self,x,y):

		self.centers 	= x
		self.values 	= y

		self.interpol 	= interp1d(x,y,
					   kind='cubic',
					   bounds_error=False,
					   fill_value=(0,0)
					   )

#---------------------------------------------------------------------------------------#
#		Load theoretical momentum distribution
#---------------------------------------------------------------------------------------#

def load_theo(file_name):
	'''Load theoretical momentum distribution from file.'''

	ppar_theo = np.loadtxt(file_name)

	return theo(ppar_theo[:,0],ppar_theo[:,1])

#---------------------------------------------------------------------------------------#
#		Convolve theoretical momentum distribution
#---------------------------------------------------------------------------------------#

def filter_p(x,limits):
	'''Create a mask (0/1) for momenta based on Atima calculations.'''

	mask = np.where((x >= limits[0])*(x <= limits[1]),
			np.ones(len(x)),np.zeros(len(x)))

	return mask

#def convolve_theo(ppar_theo,p_range,fit_res,kinematics,params,**kwargs):
def convolve_theo(ppar_theo,boost,p_range,fit_res,params,**kwargs):
	'''Convolution of theoretical momentum distributions with 
	rectangular function for uncertainty of reaction position
	and unreacted beam profile. Boost to lab frame for NSCL data.'''

	#get the x centers of the boosted distribution
	#x_centers 	= ppar_theo['boost']['x_centers']
	#values 		= ppar_theo['boost']['values']

	p				= ppar_theo['orig'].centers

	if boost:
		p *= kwargs['kinematics']['after']['gamma']

	p_sized 			= np.linspace(2*p[0],2*p[-1],2*len(p)-1)

	values 				= ppar_theo['orig'].values

	#include effect of reaction position
	p_filtered 			= filter_p(p_sized,p_range)

	values_target 			= np.convolve(values,p_filtered,mode='valid')

	ppar_theo['target'] 		= theo(p,values_target)

	#include shape of unreacted momentum distribution (p_par-p_par,0)
	#with the correction which centers it around 0
	if len(fit_res.x) == 3:
		val_unreacted	= gaussian(p_sized+fit_res.x[-2],*fit_res.x)
	elif len(fit_res.x) == 4:
		val_unreacted	= right(p_sized+fit_res.x[-2],*fit_res.x)
	elif len(fit_res.x) == 7:
		val_unreacted	= piecewise(p_sized+fit_res.x[-2],False,fit_res.x)
	else:
		val_unreacted	= piecewise(p_sized+fit_res.x[-2],True,fit_res.x)

	values_tail 			= np.convolve(values,val_unreacted,mode='valid')

	ppar_theo['tail'] 		= theo(p,values_tail)

	#include shape of unreacted momentum distribution (p_par-p_par,0)
	#and effect of reaction position
	values_tail_target 		= np.convolve(values_tail,p_filtered,mode='valid')

	ppar_theo['tail_target'] 	= theo(p,values_tail_target)

	return ppar_theo

#---------------------------------------------------------------------------------------#
#		Fit convoluted theoretical momentum distribution to data
#---------------------------------------------------------------------------------------#

def shift_scale_theo(params,x,y,scale,ppar_theo):
	'''Fit function for horizontal shift of convolved momentum distributions.'''

	x_shift 	= x - params[0]
	fit 		= scale*ppar_theo.interpol(x_shift)

	#x_shift 	= x - params[1]
	#fit 		= params[0]*ppar_theo(x_shift)

	residuals 	= np.linalg.norm(y-fit)

	return residuals

def scale_theo(spec,ppar_theo):
	'''Scale convolved momentum distributions
	vertically using the maximum bin content.'''

	#for y take the modes of the MC values
	x 	= spec.centers
	y 	= spec.mode

	#position of maximum y value for scaling
	x_max 	= x[np.argmax(y)]
	y_max 	= np.max(y)

	#scale theory to experiment
	scale 	= y_max/np.max(ppar_theo.values)

	return np.array([scale])

def fit_theo(spec,ppar_theo,fit_range):
	'''Scale convolved momentum distributions and fit them horizontally.'''

	scale 	= scale_theo(spec,ppar_theo)[0]

	#for y take the modes of the MC values
	x 	= spec.centers
	y 	= spec.mode

	#position of maximum y value for scaling
	x_max 	= x[np.argmax(y)]
	#y_max 	= np.max(y)

	#scale theory to experiment
	#scale 	= y_max/np.max(ppar_theo['values'])

	mask  	= (x >= fit_range[0])*(x <= fit_range[1])

	#cut to range
	x_val 	= spec.centers[mask]
	y_val 	= y[mask]

	res_fit_min = minimize(shift_scale_theo,
				args=(x_val,y_val,scale,ppar_theo),
				x0=x_max,
				options={'maxiter':10000})

	#res_fit_min.x = np.array([scale,res_fit_min.x[0]])

	return np.array([scale,res_fit_min.x[0]])
