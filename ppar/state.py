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

'''state class definition'''

from os import path

import numpy as np

from matplotlib.pyplot import Figure,Axes

from .ppar_RIBF import ppar_RIBF
from .ppar_NSCL import ppar_NSCL
from .spectrum import spectrum,rebin_spec,eval_spec,shift_spec
from .plots import plot_theo,plot_mode
from .theory import load_theo,convolve_theo,fit_theo,scale_theo

#---------------------------------------------------------------------------------------#
#		state class
#---------------------------------------------------------------------------------------#

class state():
	'''state class handling momentum distributions and convolutions.'''

	def __init__(self,ppar_exp: ppar_NSCL|ppar_RIBF,ppar_theo: str):
		'''
		Create state instance.

		:param ppar_exp:
			ppar_NSCl or ppar_FRIB object
		:param ppar_theo:
			file with theory
		'''

		self.params 	= {}

		if not isinstance(ppar_exp,(ppar_NSCL,ppar_RIBF)):

			raise ValueError(f'ppar_exp must be an instance of the class ppar not {type(ppar_exp)}!')

		self.ppar_exp 	= ppar_exp

		#ppar_calc
		if not path.isfile(ppar_theo):

			raise ValueError(f'file {ppar_theo} does not exist!')

		self.ppar_theo 	= {'orig':load_theo(ppar_theo)}
		#self.ppar_theo = prepare_theo(ppar_theo,ppar_exp.kin_reac)

	def convolve_theory(self,**kwargs) -> dict:
		'''
		Convolve the theoretical momentum distribution with momentum uncertainty 
		from the target and the experimental momentum distribution.

		:return ppar_theo:
			concolved theoretical momentum distributions
		'''

		#In the future store this information in params maybe?
		boost = True if isinstance(self.ppar_exp,ppar_NSCL) else False

		if boost:
			kwargs['kinematics'] = self.ppar_exp.kin_reac

		self.ppar_theo 	= convolve_theo(self.ppar_theo,
						boost,
						self.ppar_exp.p_range,
						self.ppar_exp.fit_res_unreac,
						self.ppar_exp.params,
						**kwargs
					       )

		return self.ppar_theo

	def plot_theory(self,rescale: bool=False,**kwargs) -> tuple:
		'''
		Plot the theoretical momentum distribution.

		:param rescale:
			rescale horizontal axis to GeV/c, default False 
		'''

		#return plot_theo(self.ppar_theo,self.ppar_exp.kin_reac,rescale,**kwargs)
		return plot_theo(self.ppar_theo,rescale,**kwargs)

	def rebin_hist(self,spec: spectrum,rebin: int=1,threshold: int=5):
		'''
		Rebin experimental data and shift it horizontally 
		if reacted beam has been fitted before.

		:param spec:
			experimental momentum distribution
		:param rebin:
			rebin factor for experiment
		:param threshold:
			threshold above which a bin is considered
		'''

		#spec
		if not isinstance(spec,spectrum):

			raise ValueError(f'spec must be an instance of the class spectrum not {type(spec)}!')

		#threshold below which bins are ignored
		if not isinstance(threshold,int):

			raise ValueError(f'threshold must be int not {type(threshold)}!')

		#rebin
		if not isinstance(rebin,int):

			raise ValueError(f'rebin must be int not {type(rebin)}!')

		self.params['rebin'] 	= rebin

		#shift spectrum horizontally
		try:
			self.spec	= shift_spec(spec,self.ppar_exp.fit_res_reac)

		except AttributeError:
			self.spec	= spec

		#if necessary, rebin spectrum
		if rebin > 1:
			self.spec.rebin(rebin,overwrite=True)

		self.spec.eval(threshold)

	def fit_theory(self,fit_range: list[int]) -> list[float]:
		'''
		Fit convolved theoretical momentum distribution
		to experimental data. Not recommended.

		:param fit_range:
			momentum range used for fit of experimental spectrum

		:return fit_res:
			list with vertical scaling parameter and
			horizontal displacement from fit
		'''

		#fit_range
		if isinstance(fit_range,(list,np.ndarray)) and len(fit_range) == 2:

			self.fit_range = fit_range

		else:
			raise ValueError('fit_range must be a list of length two!')

		self.fit_res = fit_theo(self.spec,self.ppar_theo['tail_target'],self.fit_range)

		return self.fit_res

	def scale_theory(self) -> list[float]:
		'''
		Scale the convolved theoretical momentum distribution
		vertically to experimental data using the maximum bin content.

		:return fit_res:
			list with vertical scaling parameter
		'''

		self.fit_res = scale_theo(self.spec,self.ppar_theo['tail_target'])

		return self.fit_res

	def plot_hist(self,plot_theory: bool=True,log: bool=False,rescale: bool=False,show_region: bool=True,**kwargs) -> tuple[Figure,Axes]:
		'''
		Plot the exclusive experimental momentum distribution 
		with fit if available.

		:param plot_theory:
			plot adjusted theory, default True
		:param log:
			toggle logarithmic vertical axis, default False
		:param rescale:
			rescale horizontal axis to GeV/c, default False
		:param show_region:
			show fit/scaling region in case fit_theory was used
		:param kwargs:
			keyword arguments passed to plot

		:return fig:
			Figure
		:return ax:
			Axes
		'''

		return plot_mode(self,self.spec,plot_theory,log,rescale,show_region,**kwargs)
