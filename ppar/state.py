'''state class definition'''

from os import path

import numpy as np

from .spectrum import rebin_spec,eval_spec,shift_spec
from .plots import plot_theo,plot_mode
from .theory import load_theo,convolve_theo,fit_theo,scale_theo

#---------------------------------------------------------------------------------------#
#		state class
#---------------------------------------------------------------------------------------#

class state():
	'''state class handling momentum distributions and convolutions.'''

	def __init__(self,ppar_exp,ppar_theo):
		'''Create state instance.'''

		self.params = {}

		if not isinstance(ppar_exp,object) or ppar_exp.__class__.__name__ not in ['ppar_NSCL','ppar_RIBF']:

			raise ValueError(f'ppar_exp must be an instance of the class ppar not {type(ppar_exp)}!')

		self.ppar_exp = ppar_exp

		#ppar_calc
		if not path.isfile(ppar_theo):

			raise ValueError(f'file {ppar_theo} does not exist!')

		self.ppar_theo = load_theo(ppar_theo)
		#self.ppar_theo = prepare_theo(ppar_theo,ppar_exp.kin_reac)

	def convolve_theory(self,**kwargs):
		'''Convolve the theoretical momentum distribution with
		momentum uncertainty from the target and
		the experimental momentum distribution.'''

		#In the future store this information in params maybe?
		boost = True if self.ppar_exp.__class__.__name__ == 'ppar_NSCL' else False

		if boost:
			kwargs['kinematics'] = self.ppar_exp.kin_reac

		#self.ppar_theo = convolve_theo(self.ppar_theo,self.ppar_exp.p_range,
		#				self.ppar_exp.fit_res_unreac,self.ppar_exp.kin_reac,
		#				self.ppar_exp.params,**kwargs)
		self.ppar_theo = convolve_theo(self.ppar_theo,
						boost,
						self.ppar_exp.p_range,
						self.ppar_exp.fit_res_unreac,
						self.ppar_exp.params,
						**kwargs
					       )

		return self.ppar_theo

	def plot_theory(self,rescale=False,**kwargs):
		'''Plot the theoretical momentum distribution.'''

		#return plot_theo(self.ppar_theo,self.ppar_exp.kin_reac,rescale,**kwargs)
		return plot_theo(self.ppar_theo,rescale,**kwargs)

	def rebin_hist(self,spec,rebin=1,threshold=5):
		'''Rebin experimental data and shift it horizontally 
		if reacted beam has been fitted before.'''

		#spec
		if not isinstance(spec,dict):

			raise ValueError(f'spec {spec} must be dict with keys x_centers and values!')

		#threshold below which bins are ignored
		if not isinstance(threshold,int):

			raise ValueError(f'threshold must be int not {type(threshold)}!')

		#rebin
		if not isinstance(rebin,int):

			raise ValueError(f'rebin must be int not {type(rebin)}!')

		self.params['rebin'] 	= rebin

		#shift spectrum horizontally
		try:
			spec_shifted	= shift_spec(spec,self.ppar_exp.fit_res_reac)
		
		except AttributeError:
			pass

		self.spec  		= eval_spec(spec_shifted,threshold)

		#if necessary, rebin spectrum
		if rebin > 1:
			
			self.rebin 	= rebin_spec(spec_shifted,rebin)
			self.rebin 	= eval_spec(self.rebin,threshold)
		
		else:
			self.rebin 	= self.spec

	def fit_theory(self,fit_range):
		'''Fit convolved theoretical momentum distribution
		to experimental data. Not recommended.'''

		#fit_range
		if isinstance(fit_range,(list,np.ndarray)) and len(fit_range) == 2:

			self.fit_range = fit_range

		else:
			raise ValueError('fit_range must be a list of length two!')

		self.fit_res = fit_theo(self.rebin,self.ppar_theo['tail_target'],self.fit_range)

		return self.fit_res

	def scale_theory(self):
		'''Scale the convolved theoretical momentum distribution
		vertically to experimental data using the maximum bin content.'''

		self.fit_res = scale_theo(self.rebin,self.ppar_theo['tail_target'])

		return self.fit_res

	def plot_hist(self,rebin=True,plot_theory=True,log=False,rescale=False,show_region=True,**kwargs):
		'''Plot the exclusive experimental momentum distribution with fit if available.'''

		#rebin
		if not isinstance(rebin,bool):

			raise ValueError(f'rebin must be bool not {type(rebin)}!')

		if rebin:

			return plot_mode(self,self.rebin,plot_theory,log,rescale,show_region,**kwargs)

		return plot_mode(self,self.spec,plot_theory,log,rescale,show_region,**kwargs)
