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

	def __init__(self,beam,target,product,Brho_reac,Brho_unreac,verbose=True):
		'''Create ppar instance for NSCL data.'''

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

	def calc_stopping(self,threshold=5,method='bw'):
		'''Calculate stopping in the target using Atima.'''

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

	def plot_stopping(self,**kwargs):
		'''Plot results of stopping calculation using Atima.'''

		return plot_stop(self.stopping['data'],self.stopping['interpol'],
				self.beam,self.product,**kwargs)

	def load_unreacted(self,file,histogram,rebin=1):
		'''Load the experimental histogram of unreacted beam setting and extract data.'''

		#file path
		if not path.isfile(file):

			raise ValueError(f'file {file} does not exist!')

		#rebin
		if not isinstance(rebin,int):

			raise ValueError(f'rebin must be int not {type(rebin)}!')

		#self.spec_unreac	= prepare_spec(file,histogram,self.kin_reac,
			#					self.beam,self.product)
		self.spec_unreac  	= load_file(file,histogram)
		self.rebin_unreac 	= rebin_spec(self.spec_unreac,rebin)

		#add identifiers
		#self.spec_unreac['reac']	= False
		#self.rebin_unreac['reac']	= False

	def load_reacted(self,file,histogram,rebin=1):
		'''Load the experimental histogram of reaction setting and extract data.'''

		#file path
		if not path.isfile(file):

			raise ValueError(f'file {file} does not exist!')

		#rebin
		if not isinstance(rebin,int):

			raise ValueError(f'rebin must be int not {type(rebin)}!')

		self.spec_reac  	= load_file(file,histogram)
		self.rebin_reac 	= rebin_spec(self.spec_reac,rebin)

		#add identifiers
		#self.spec_unreac['reac']	= True
		#self.rebin_unreac['reac']	= True

	def fit_unreacted(self,fit_range,x0):
		'''Fit the experimental histogram to extract its functional shape.'''

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

	def fit_reacted(self,fit_range,x0):
		'''Fit the experimental histogram to extract its functional shape.'''

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

	def plot_unreacted(self,rebin=True,plot_fit=True,log=False,rescale=False,**kwargs):
		'''Plot the experimental momentum distribution with fit if available.'''

		#rebin
		if not isinstance(rebin,bool):

			raise ValueError(f'rebin must be bool not {type(rebin)}!')

		#log
		if not isinstance(log,bool):

			raise ValueError(f'log must be bool not {type(log)}!')

		if rebin:

			return plot_ppar(self,self.rebin_unreac,plot_fit,log,rescale,False,**kwargs)

		return plot_ppar(self,self.spec_unreac,plot_fit,log,rescale,False,**kwargs)

	def plot_reacted(self,rebin=True,plot_fit=True,log=False,rescale=False,**kwargs):
		'''Plot the experimental momentum distribution with fit if available.'''

		#rebin
		if not isinstance(rebin,bool):

			raise ValueError(f'rebin must be bool not {type(rebin)}!')

		#log
		if not isinstance(log,bool):

			raise ValueError(f'log must be bool not {type(log)}!')

		if rebin:

			return plot_ppar(self,self.rebin_reac,plot_fit,log,rescale,True,**kwargs)

		return plot_ppar(self,self.spec_reac,plot_fit,log,rescale,True,**kwargs)

