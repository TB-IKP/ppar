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

'''ppar class definition for NSCL/FRIB data'''

from os import path

import numpy as np

from scipy.optimize import OptimizeResult
from matplotlib.pyplot import Figure,Axes

from .spectrum import rebin_spec,fit_spec,load_file,eval_spec
from .plots import plot_ppar,plot_stop
from .kinematics import kinematics_reaction,stopping_target_fw,momentum_range_fw
from .kinematics import stopping_target_bw,momentum_range_bw,correct_kinematics
from .utils import mg_cm2_to_um,atomic_mass_to_ion_mass,extract_Brho
from .messages import header_message,start_message,nucl_message,kinematics_message

#---------------------------------------------------------------------------------------#
#	NSCL:	ppar class
#---------------------------------------------------------------------------------------#

class ppar_NSCL:
	'''Class for analysis of parallel momentum distributions from NSCL/FRIB data.'''

	def __init__(self,beam: dict,target: dict,product: dict,Brho_reac: str|dict,Brho_unreac: str|dict,verbose: bool=True):
		'''
		Create ppar instance for NSCL data.

		:param beam:
			beam details with keys name, A, Z, mass
		:param target:
			target details with keys name, A, Z, mass, 
			thickness, density
		:param product:
			product details with keys name, A, Z, mass 
		:param Brho_reac:
			path to Barney file for reaction setting or
			dict with rigidities with keys ``before`` and
			``after`` for BTS33 (Seg 7) and BTS34 (Seg 8)
		:param Brho_unreac:
			path to Barney file for unreacted setting or
			dict with rigidities with keys ``before`` and
			``after`` for BTS33 (Seg 7) and BTS34 (Seg 8)
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

			start_message('NSCL')

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
		if verbose:

			header_message('Kinematics')

		#Brho
		#reaction setting
		if isinstance(Brho_reac,str):

			if not path.isfile(Brho_reac):

				raise ValueError(f'file {Brho_reac} does not exist!')

			Brho_reac 	= {key:extract_Brho(Brho_reac,key) for key in ['before','after']}

		elif not isinstance(Brho_reac,dict):

			raise ValueError('Brho_reac must either be path to Barney file or dict with keys before and after!')

		#beam in, product out
		#self.kin_reac 	= kinematics_reaction(Brho_reac,[self.beam,self.product])
		self.kin_reac 	= {'before':kinematics_reaction(Brho_reac['before'],self.beam),
				   'after': kinematics_reaction(Brho_reac['after'], self.product)
				  }

		#reaction message beam in, product out
		if verbose:

			kinematics_message(self.kin_reac,self.beam,self.target,self.product)

		#unreacted beam setting
		if isinstance(Brho_unreac,str):

			if not path.isfile(Brho_unreac):

				raise ValueError(f'file {Brho_unreac} does not exist!')

			Brho_unreac 	= {key:extract_Brho(Brho_unreac,key) for key in ['before','after']}

		elif not isinstance(Brho_unreac,dict):

			raise ValueError('Brho_unreac must either be path to Barney file or dict with keys before and after!')

		#beam in, beam out
		#self.kin_unreac = kinematics_reaction(Brho_unreac,[self.beam,self.beam])
		self.kin_unreac = {'before':kinematics_reaction(Brho_unreac['before'],self.beam),
				   'after': kinematics_reaction(Brho_unreac['after'], self.beam)
				  }

		#reaction message beam in, beam out
		if verbose:

			kinematics_message(self.kin_unreac,self.beam,self.target,self.beam)

	def calc_stopping(self,threshold: int=5,method: str='bw'):
		'''
		Calculate stopping in the target using Atima.

		:param threshold:
			termination criterion for stopping calculation
		:param method:
			``fw`` (from BTS33) or ``bw`` (from BTS34) calculation of stopping
		'''

		#threshold
		if not isinstance(threshold,(int,float)):

			raise ValueError(f'threshold must be int not {type(threshold)}!')

		self.params['threshold'] = int(threshold)

		#method
		if method in ['bw','backward','backwards']:

			#momentum before target for two cases:
			#1.) reaction happens at the beginning of the target
			#2.) reaction happens at the end of the target

			self.stopping 	= stopping_target_bw(self.kin_reac,self.params['threshold'],
							self.beam,self.target,self.product,
							self.params['verbose'])
			self.p_range 	= momentum_range_bw(self.kin_reac,self.stopping,
							self.beam,self.target,self.product)

		elif method in ['fw','forward']:

			#momentum after target for two cases:
			#1.) reaction happens at the beginning of the target
			#2.) reaction happens at the end of the target

			self.stopping 	= stopping_target_fw(self.kin_reac,self.params['threshold'],
							self.beam,self.target,self.product,
							self.params['verbose'])
			self.p_range 	= momentum_range_fw(self.kin_reac,self.stopping,
							self.beam,self.target,self.product)

		else:
			raise ValueError(f'method must be bw of fw not {method}!')

	def plot_stopping(self,**kwargs) -> tuple:
		'''Plot results of stopping calculation using Atima.'''

		return plot_stop(self.stopping['data'],self.stopping['interpol'],
				self.beam,self.product,**kwargs)

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

		#self.spec_unreac	= prepare_spec(file,histogram,self.kin_reac,
			#					self.beam,self.product)
		self.spec_unreac  	= load_file(file,hist)
		self.rebin_unreac 	= rebin_spec(self.spec_unreac,rebin)

		#add identifiers
		#self.spec_unreac['reac']	= False
		#self.rebin_unreac['reac']	= False

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

		self.spec_reac  	= load_file(file,hist)
		self.rebin_reac 	= rebin_spec(self.spec_reac,rebin)

		#add identifiers
		#self.spec_unreac['reac']	= True
		#self.rebin_unreac['reac']	= True

	def fit_unreacted(self,fit_range: list[int],x0: list[float]) -> OptimizeResult:
		'''
		Fit the experimental histogram of unreacted beam to extract its functional shape.

		:param fit_range:	
			range for spectrum fit
		:param x0:
			initial parameter guess, length specifies fit function
			3 	Gaussian
			4 	two erf functions
			7 	two erf functions with exponential tail
			10 	two erf functions with exponential tail and Gaussian

		:return fit_res_unreac:
			fit result
		'''

		#fit_range
		if isinstance(fit_range,(list,np.ndarray)) and len(fit_range) == 2:

			self.fit_range_unreac = fit_range

		else:
			raise ValueError('fit_range must be a list of length two!')

		#x0
		if not isinstance(fit_range,(list,np.ndarray)):

			raise ValueError('x0 must be a list containing the start parameters!')

		#if len(x0) == 7:

		#	self.params['gauss'] 	= False

		#elif len(x0) == 10:

		#	self.params['gauss'] 	= True

		#else:
		#	raise ValueError('x0 must be a list of length ten(seven) if a Gaussian is (not) included!')

		if len(x0) not in [3,4,7,10]:

			raise ValueError('x0 must be list of length three for a Gaussian fit, \
				four for two error functions, seven for an exponential tail, \
				and ten for a combination of exponential and Gaussian tail!')

		#self.fit_res_unreac 		= fit_spec(self.rebin_unreac,self.fit_range_unreac,
		#						x0,self.params['gauss'])
		self.fit_res_unreac 		= fit_spec(self.rebin_unreac,self.fit_range_unreac,x0)

		#correct kinematics unreacted run
		self.kin_unreac['after'] 	= correct_kinematics(self.kin_unreac['after']['p'],
								self.fit_res_unreac,self.beam)

		return self.fit_res_unreac

	def fit_reacted(self,fit_range: list[int],x0: list[float]) -> OptimizeResult:
		'''
		Fit the experimental histogram from reaction setting to extract its functional shape.

		:param fit_range:	
			range for spectrum fit
		:param x0:
			initial parameter guess, length specifies fit function
			3 	Gaussian
			4 	two erf functions
			7 	two erf functions with exponential tail
			10 	two erf functions with exponential tail and Gaussian

		:return fit_res_reac:
			fit result
		'''

		#fit_range
		if isinstance(fit_range,(list,np.ndarray)) and len(fit_range) == 2:

			self.fit_range_reac = fit_range

		else:
			raise ValueError('fit_range must be a list of length two!')

		#x0
		#if not isinstance(fit_range,(list,np.ndarray)) or len(x0) != 4:

		#	raise ValueError('x0 must be a list of length four containing the start parameters!')

		#self.fit_res_reac 		= fit_peak(self.rebin_reac,self.fit_range_reac,x0)

		if len(x0) not in [3,4,7,10]:

			raise ValueError('x0 must be list of length three for a Gaussian fit, \
				four for two error functions, seven for an exponential tail, \
				and ten for a combination of exponential and Gaussian tail!')

		self.fit_res_reac 		= fit_spec(self.rebin_reac,self.fit_range_reac,x0)

		#correct kinematics unreacted run
		self.kin_reac['after'] 		= correct_kinematics(self.kin_reac['after']['p'],
								self.fit_res_reac,self.product)

		return self.fit_res_reac

	def plot_unreacted(self,rebin: bool=True,plot_fit: bool=True,log: bool=False,rescale: bool=False,**kwargs) -> tuple[Figure,Axes]:
		'''
		Plot the experimental momentum distribution with fit if available.

		:param rebin:
			plot rebinned spectrum, default True
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

		#rebin
		if not isinstance(rebin,bool):

			raise ValueError(f'rebin must be bool not {type(rebin)}!')

		#log
		if not isinstance(log,bool):

			raise ValueError(f'log must be bool not {type(log)}!')

		if rebin:

			return plot_ppar(self,self.rebin_unreac,plot_fit,log,rescale,False,**kwargs)

		return plot_ppar(self,self.spec_unreac,plot_fit,log,rescale,False,**kwargs)

	def plot_reacted(self,rebin: bool=True,plot_fit: bool=True,log: bool=False,rescale: bool=False,**kwargs) -> tuple[Figure,Axes]:
		'''
		Plot the experimental momentum distribution with fit if available.

		:param rebin:
			plot rebinned spectrum, default True
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

		#rebin
		if not isinstance(rebin,bool):

			raise ValueError(f'rebin must be bool not {type(rebin)}!')

		#log
		if not isinstance(log,bool):

			raise ValueError(f'log must be bool not {type(log)}!')

		if rebin:

			return plot_ppar(self,self.rebin_reac,plot_fit,log,rescale,True,**kwargs)

		return plot_ppar(self,self.spec_reac,plot_fit,log,rescale,True,**kwargs)

