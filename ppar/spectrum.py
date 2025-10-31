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

'''Operations on 1d spectra and fit'''

from copy import deepcopy

import re
import uproot

import numpy as np

from numpy.random import poisson
from scipy.optimize import minimize,fmin
from matplotlib.pyplot import Figure,Axes

#from .utils import gaussian,left,right,piecewise
from .utils import calc_mode,calc_sc

#---------------------------------------------------------------------------------------#
#		spectrum class
#---------------------------------------------------------------------------------------#

class spectrum():
	'''spectrum class'''

	def __init__(self,file_name: str,hist_name: str,**kwargs):
		'''
		Create spectrum instance for experimental data.

		:param file_name:	
			.root file with spectrum
		:param hist_name:
			name of histogram in file
		'''

		self.file_name 	= file_name
		self.hist_name 	= hist_name

		self.centers,self.edges,self.values = \
			load_file(file_name,hist_name,**kwargs)

		self.ndim 	= self.values.ndim
		self.bins 	= len(self.centers)

		if self.ndim == 2:
			_,self.mc_num 	= self.values.shape

	def rebin(self,rebin: int,overwrite: bool=True):
		'''
		Rebin spectrum.

		:param rebin:
			rebin factor
		:param overwrite:
			overwrite loaded data, default True
			if False, _rebin is used as suffix
		'''

		if overwrite:
			self.centers,self.edges,self.values = \
				rebin_spec(self.centers,self.edges,self.values,rebin)
		else:
			self.centers_rebin,self.edges_rebin,self.values_rebin = \
				rebin_spec(self.centers,self.edges,self.values,rebin)

	def eval(self,threshold: int=0):
		'''
		Evaluate Monte-Carlo sampled spectrum in terms of 
		mode and shoprtest-coverage interval for bins above threshold.

		:param threshold:
			threshold for which a bin 
			is included in ind_nonzero
		'''

		if self.ndim == 2:
		
			self.ind_nonzero,self.mode,self.sc_interval = \
				eval_spec(self.centers,self.edges,self.values,threshold)

	def plot(self,log: bool=False,rescale: bool=False,**kwargs) -> tuple[Figure,Axes]:
		'''
		Plot spectrum.

		:param log:
			use logarithmic vertical scale, default False
		:param rescale:
			rescale horizontal axis to GeV/c, default False

		:return fig:
			Figure
		:return ax:
			Axes
		'''

		from .plots import plot_hist

		return plot_hist(self,log,rescale,**kwargs)

	def copy(self):
		'''
		Create a deepcopy of the spectrum without results
		from eval method (mode, sc_interval, ind_nonzero).

		:return spectrum:
			deepcopy of spectrum object
		'''

		_copy = deepcopy(self)

		for attr in ['ind_nonzero','mode','sc_interval']:
			
			try:
				delattr(_copy,attr)

			except AttributeError:
				continue

		return _copy

#---------------------------------------------------------------------------------------#
#		Load data and isolate histogram data
#---------------------------------------------------------------------------------------#

def load_file(file_name,hist_name,**kwargs):
	'''Load .root files and extract histogram data.'''

	with uproot.open(file_name) as file:

		try:
			_hist_name 	= re.findall(hist_name,'|'.join(file.classnames().keys()))[0]

		except IndexError:

			hist_names 	= [hist for hist in file.classnames().keys() if 'GVALUE' not in hist]

			raise FileNotFoundError(f'histogram name {hist_name} not found \
						in {hist_names}!')

		xaxis		= file[_hist_name].all_members['fXaxis']
		values		= file[_hist_name].values()

		try:
			values	= np.array([poisson(i,int(kwargs['mc_num'])) \
						for i in values]).astype('float64')

		except KeyError:
			pass

		return xaxis.centers(),xaxis.edges(),values

def prepare_spec(file_name,hist_name,kinematics,beam,product):
	'''Obtain and prepare histogram data.'''

	#Load data
	data 			= load_file(file_name,hist_name)

	#Account for nucleons lost in reaction
	corr_fac		= product['ion mass']/beam['ion mass']

	data['x_centers']      *= corr_fac
	data['x_edges']	       *= corr_fac

	return data

#---------------------------------------------------------------------------------------#
#		Shift histogram horizontally
#---------------------------------------------------------------------------------------#

def shift_spec(spec,params,func):

	spec_shifted = deepcopy(spec)

	shift = fmin(lambda x: -func(x,*params.x),0,disp=0)

	spec_shifted.centers -= shift
	spec_shifted.edges   -= shift

	#spec_shifted.centers -= params.x[-2]
	#spec_shifted.edges   -= params.x[-2]

	return spec_shifted

#---------------------------------------------------------------------------------------#
#		Rebin histogram
#---------------------------------------------------------------------------------------#

def rebin_spec(centers,edges,values,rebin):
	'''Rebin histogram.'''

	if not isinstance(rebin,int) or rebin < 0:

		raise ValueError(f'rebin must be a positive-valued integer!')

	if rebin == 1:

		return centers,edges,values

	bins 	= len(centers)

	if bins % rebin:

		raise ValueError(f'number of bins ({bins}) must be an integer multiple \n \
				of the rebin factor {rebin}!')

	if values.ndim == 1:

		_values 	= np.sum(values.reshape(int(bins/rebin),rebin),axis=1)

	elif values.ndim == 2:

		_,mc_num 	= values.shape
		_values 	= np.sum(values.reshape(int(bins/rebin),rebin,mc_num),axis=1)

	else:
		raise ValueError(f'dimension of values ({values.ndim}) must be one or two!')

	#edges and centers
	_centers	= centers if rebin % 2 else edges

	return _centers[int(rebin/2)::rebin],edges[::rebin],_values

#---------------------------------------------------------------------------------------#
#		Evaluate histogram
#---------------------------------------------------------------------------------------#

def eval_spec(centers,edges,values,threshold):
	'''Calculate mode and shortest coverage interval for each bin.'''

	len_spec 	= len(centers)

	ind_nonzero 	= []
	mode		= np.zeros(len_spec)
	sc_interval 	= np.zeros((len_spec,2))

	for i,j in enumerate(values):

		if not np.all(j==0) and np.mean(j) > threshold:
		#if np.count_nonzero(j) > np.sqrt(mc_num) and np.mean(j) > threshold:

			ind_nonzero.append(i)

			mode[i]        = calc_mode(j)
			sc_interval[i] = calc_sc(j)

	return ind_nonzero,mode,sc_interval

#---------------------------------------------------------------------------------------#
#		Fit histogram
#---------------------------------------------------------------------------------------#

# def fit_func(x):
# 	'''Compose the fit function for the 
# 	experimental momentum distributions.'''

# 	#Gaussian
# 	if len(x0) == 3:

# 		return gaussian

# 	#two erf functions
# 	elif len(x0) == 4:

# 		return right

# 	#Gaussian with exponential tail
# 	elif len(x0) == 5:

# 		return 

# 	#two erf functions with exponential tail
# 	#with (10) and without (7) Gaussian
# 	else:
# 		return piecewise

# def piecewise_minimize(params,x,y,gauss):
# 	'''Fit residuals of piecewise function with penalty term for continuity.'''

# 	#args[0] = switch

# 	#left
# 	#args[1] = exp1
# 	#args[2] = exp2
# 	#args[3] = area
# 	#args[4] = mean
# 	#args[5] = sigma

# 	#right
# 	#args[6] = height
# 	#args[7] = width
# 	#args[8] = center
# 	#args[9] = sigma

# 	#add a shift in index
# 	shift = 3 if gauss else 0

# 	fit 		= piecewise(x,gauss,params)

# 	residuals 	= np.linalg.norm(y-fit)
# 	penalty   	= np.linalg.norm(left(params[0],gauss,*params[1:3+shift])-\
# 				   	 right(params[0],*params[3+shift:]))

# 	return residuals + penalty

# def right_minimize(params,x,y):
# 	'''Fit residuals of asymmetric peak.'''

# 	fit		= right(x,*params)
# 	residuals 	= np.linalg.norm(y-fit)

# 	return residuals

# def gaussian_minimize(params,x,y):
# 	'''Fit residuals of Gaussian peak.'''

# 	fit		= gaussian(x,*params)
# 	residuals 	= np.linalg.norm(y-fit)

# 	return residuals

def func_minimize(params,x,y,func):
	'''Fit residuals for supplied function.'''

	fit 		= func(x,*params)
	residuals 	= np.linalg.norm(y-fit)

	return residuals

def fit_spec(spec,fit_range,x0,func,minimizer):
	'''Fit of experimental momentum distributions with low-momentum tail.'''

	mask  	= (spec.centers >= fit_range[0])*\
		  (spec.centers <= fit_range[1])

	x_val 	= spec.centers[mask]
	y_val 	= spec.values[mask]

	opt_res_min 	= minimize(minimizer,
				args=(x_val,y_val,func),
				x0=x0,
				method='Nelder-Mead',
				options={'maxiter':10000})

	return opt_res_min
