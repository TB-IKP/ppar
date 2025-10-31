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

'''ppar class definition for RIBF data'''

from os import path
from typing import Callable

import numpy as np

from scipy.optimize import OptimizeResult
from matplotlib.pyplot import Figure,Axes

from .spectrum import spectrum,fit_spec,func_minimize
from .plots import plot_ppar
from .utils import mg_cm2_to_um,atomic_mass_to_ion_mass,gaussian
from .messages import start_message,nucl_message

#---------------------------------------------------------------------------------------#
#	RIBF:	ppar class
#---------------------------------------------------------------------------------------#

class ppar_RIBF():
	'''Class for analysis of parallel momentum distributions from RIBF data.'''

	def __init__(self,beam: dict,target: dict,product: dict,verbose: bool=True):
		'''
		Create ppar instance for RIBF data.

		:param beam:
			beam details with keys name, A, Z, mass
		:param target:
			target details with keys name, A, Z, mass, 
			thickness, density
		:param product:
			product details with keys name, A, Z, mass 
		:param verbose:
			toggle messages, default True
		'''

		#parameters
		self.params = {}

		#verbosity level
		if isinstance(verbose,bool):

			self.params['verbose'] = verbose

		else:
			raise ValueError(f'verbose must be bool not {type(verbose)}!')

		#start message
		if verbose:

			start_message('RIBF')

		#beam
		if isinstance(beam,dict):

			self.beam 	= atomic_mass_to_ion_mass(beam)

		else:
			raise ValueError('beam must be dict with keys Z and A!')

		#target
		if isinstance(target,dict):

			self.target 	= atomic_mass_to_ion_mass(target)
			self.target 	= mg_cm2_to_um(self.target)

		else:
			raise ValueError('target must be dict with keys Z, A,\n \
					thickness (in mg/cm2) and density!')

		#product
		if isinstance(product,dict):

			self.product 	= atomic_mass_to_ion_mass(product)

		else:
			raise ValueError('product must be dict with keys Z and A!')

		#nucl message
		if verbose:

			nucl_message(self.beam,self.target,self.product)

		#header kinematics
		#if verbose:

		#	header_message('Kinematics')

	def calc_stopping(self):
		'''Calculate stopping in the target using Atima.'''

		raise NotImplementedError('Not yet implemented! Use set_p_range method instead!')

		try:
			import atima

		except ModuleNotFoundError:
			print('Module atima not found. Use set_p_range method instead!')

		# #incoming energy (BigRIPS)
		# if not isinstance(en_in,(int,float)):

		# 	raise ValueError(f'en_in must be float not {type(en_in)}!')

		# #outgoing energy (ZDS)
		# if not isinstance(en_out,(int,float)):

		# 	raise ValueError(f'en_out must be float not {type(en_out)}!')

		# #maybe hand over here spectra and determine limits automatically...

		# #steps
		# if not isinstance(steps,(int,float)):

		# 	raise ValueError(f'steps must be int not {type(steps)}!')

		# self.stopping 	= stopping_target(en_in,en_out,steps,
		# 			self.beam,self.target,self.product)

		# self.p_range 	= momentum_range(self.stopping,
		# 			self.beam,self.target,self.product)

	def plot_stopping(self,**kwargs):
		'''Plot results of stopping calculation using Atima.'''

		raise NotImplementedError('Not yet implemented!')

	# 	return plot_stop(self.stopping['data'],self.stopping['interpol'],
	# 			self.beam,self.product,**kwargs)

	def set_p_range(self,p_range: list[float]):
		'''
		Insert momentum uncertainty due to reaction position along the target.
		Will be replaced with actual calculation later (as for NSCL data).

		:param p_range:
			Symmetrized momentum spread, format [-p,p]
		'''

		if isinstance(p_range,(list,np.ndarray)) and len(p_range) == 2:

			self.p_range = p_range

		else:
			raise ValueError('p_range must be a list of length two!')

	def load_unreacted(self,file: str,hist: str,rebin: int=1):
		'''
		Load the experimental histogram of unreacted beam setting and extract data.

		:param file:
			.root file with spectrum
		:param hist:
			name of histogram in file
		:param rebin:
			rebin factor, default 1
		'''

		#file path
		if not path.isfile(file):

			raise ValueError(f'file {file} does not exist!')

		#rebin
		if not isinstance(rebin,int):

			raise ValueError(f'rebin must be int not {type(rebin)}!')

		self.spec_unreac  	= spectrum(file,hist)

		if rebin > 1:
			self.spec_unreac.rebin(rebin,overwrite=True)

	def load_reacted(self,file: str,hist: str,rebin: int=1):
		'''
		Load the experimental histogram of reaction setting and extract data.

		:param file:
			.root file with spectrum
		:param hist:
			name of histogram in file
		:param rebin:
			rebin factor, default 1
		'''

		#file path
		if not path.isfile(file):

			raise ValueError(f'file {file} does not exist!')

		#rebin
		if not isinstance(rebin,int):

			raise ValueError(f'rebin must be int not {type(rebin)}!')

		self.spec_reac  	= spectrum(file,hist)

		if rebin > 1:
			self.spec_reac.rebin(rebin,overwrite=True)

	def fit_unreacted(self,fit_range: list[int],x0: list[float]=[1e6,0,100],
				func: Callable=gaussian, minimizer: Callable=func_minimize) -> OptimizeResult:
		'''
		Fit the experimental histogram of unreacted beam to extract its functional shape.

		:param fit_range:	
			range for spectrum fit
		:param x0:
			initial parameter guess,
			default 1e6,0,100]
		:param func:
			externally defined fit function, 
			default gaussian
		:param minimizer:
			externally defined minimizer for fit,
			default ||y-f(x)||_2

		:return fit_res_unreac:
			fit result
		'''

		#fit_range
		if isinstance(fit_range,(list,np.ndarray)) and len(fit_range) == 2:

			self.fit_range_unreac = fit_range

		else:
			raise ValueError('fit_range must be a list of length two!')

		#x0
		if not isinstance(x0,(list,np.ndarray)):

			raise ValueError('x0 must be a list containing the start parameters!')

		#func
		if callable(func):

			print(f'Using fit function {func.__name__}.')

			self.fit_func_unreac 	= func

		else:
			raise ValueError('func must be callable!')

		#minimizer
		if callable(minimizer):

			print(f'Using minimizer {minimizer.__name__}.')

		else:
			raise ValueError('minimizer must be callable!')
		
		self.fit_res_unreac 		= fit_spec(self.spec_unreac,self.fit_range_unreac,
								x0,self.fit_func_unreac,minimizer)

		#correct kinematics unreacted run
		#self.kin_unreac['after'] 	= correct_kinematics(self.kin_unreac['after']['p'],
		#						self.fit_res_unreac,self.beam)

		return self.fit_res_unreac

	def fit_reacted(self,fit_range: list[int],x0: list[float]=[1e6,0,100],
				func: Callable=gaussian, minimizer: Callable=func_minimize) -> OptimizeResult:
		'''
		Fit the experimental histogram from reaction setting to extract its functional shape.

		:param fit_range:	
			range for spectrum fit
		:param x0:
			initial parameter guess,
			default 1e6,0,100]
		:param func:
			externally defined fit function, 
			default gaussian
		:param minimizer:
			externally defined minimizer for fit,
			default ||y-f(x)||_2

		:return fit_res_reac:
			fit result
		'''

		#fit_range
		if isinstance(fit_range,(list,np.ndarray)) and len(fit_range) == 2:

			self.fit_range_reac = fit_range

		else:
			raise ValueError('fit_range must be a list of length two!')

		#x0
		if not isinstance(x0,(list,np.ndarray)):

			raise ValueError('x0 must be a list containing the start parameters!')

		#func
		if callable(func):

			print(f'Using fit function {func.__name__}.')

			self.fit_func_reac 	= func

		else:
			raise ValueError('func must be callable!')

		#minimizer
		if callable(minimizer):

			print(f'Using minimizer {minimizer.__name__}.')

		else:
			raise ValueError('minimizer must be callable!')

		self.fit_res_reac 		= fit_spec(self.spec_reac,self.fit_range_reac,
								x0,self.fit_func_reac,minimizer)

		#correct kinematics unreacted run
		#self.kin_reac['after'] 	= correct_kinematics(self.kin_reac['after']['p'],
		#						self.fit_res_reac,self.beam)

		return self.fit_res_reac

	def plot_unreacted(self,plot_fit: bool=True,log: bool=False,
				rescale: bool=False,**kwargs) -> tuple[Figure,Axes]:
		'''
		Plot the experimental momentum distribution with fit if available.

		:param plot_fit:
			plot fit if available, default True
		:param log:
			toggle logarithmic vertical axis, default False
		:param rescale:
			rescale horizontal axis to GeV/c, default False
		:param kwargs:
			keyword arguments passed to plot

		:return fig:
			Figure
		:return ax:
			Axes
		'''

		#log
		if not isinstance(log,bool):

			raise ValueError(f'log must be bool not {type(log)}!')

		return plot_ppar(self,self.spec_unreac,plot_fit,log,rescale,False,**kwargs)

	def plot_reacted(self,plot_fit: bool=True,log: bool=False,
				rescale: bool=False,**kwargs) -> tuple[Figure,Axes]:
		'''
		Plot the experimental momentum distribution with fit if available.

		:param plot_fit:
			plot fit if available, default True
		:param log:
			toggle logarithmic vertical axis, default False
		:param rescale:
			rescale horizontal axis to GeV/c, default False
		:param kwargs:
			keyword arguments passed to plot

		:return fig:
			Figure
		:return ax:
			Axes
		'''

		#log
		if not isinstance(log,bool):

			raise ValueError(f'log must be bool not {type(log)}!')

		return plot_ppar(self,self.spec_reac,plot_fit,log,rescale,True,**kwargs)
