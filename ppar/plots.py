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

'''Plots for ppar'''

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import Figure,Axes
from matplotlib.ticker import MaxNLocator,LinearLocator
from scipy.optimize import fmin

from .spectrum import spectrum,eval_spec
from .utils import name_nucl#,gaussian,right,piecewise

#---------------------------------------------------------------------------------------#
#		Settings
#---------------------------------------------------------------------------------------#

#Colors are used in plot of stopping calculations
Dic_colors = {'beam':'black','product':'orange','target':'orange'}

#---------------------------------------------------------------------------------------#
#		Plot momentum distribution
#---------------------------------------------------------------------------------------#

def plot_hist(spec: spectrum,log: bool=False,rescale: bool=False,**kwargs) -> tuple[Figure,Axes]:
	'''
	Plot experimental momentum distribution.

	:param spec:
		spectrum object with experimental data
	:param log:
		use logarithmic vertical scale, default False
	:param rescale:
		rescale horizontal axis to GeV/c, default False

	:return fig:
		Figure
	:return ax:
		Axes
	'''

	#A few settings are overwritten or re-done if the function 
	#is called from plot_ppar (e.g. NSCL and RIBF differences).
	#Some options exist in here to make this function usuable
	#if called from outside.

	if not isinstance(spec,spectrum):

		raise ValueError(f'spec must be an instance of the class spectrum not {type(spec)}!')

	#Convert p to GeV/c
	scale = 1e-3 if rescale else 1

	fig,ax = plt.subplots(figsize=(10,4))
	fig.subplots_adjust(bottom=0.2)

	try:
		label = kwargs['label']
	except KeyError:
		label ='Experiment'

	if spec.values.ndim == 1:

		ax.step(spec.centers*scale,
			spec.values,
			where='mid',
			color='black',
			label=label,
			#label=kwargs['label'],
			)

	else:
		if not hasattr(spec,'ind_nonzero'):
			spec.eval(0)

		ax.step(spec.centers*scale,
			spec.mode,
			where='mid',
			color='black',
			label=label,
			#label=kwargs['label'],
			)

		ax.fill_between(
			spec.centers,
			spec.sc_interval[:,0],
			spec.sc_interval[:,1],
			step='mid',
			color='grey',
			alpha=0.5
			)

	if log:
		ax.set_yscale('log')

	#plot limits from kwargs
	for key in kwargs:

		if key == 'xlim':
			ax.set_xlim(np.min(kwargs['xlim']),np.max(kwargs['xlim']))
		elif key == 'ylim':
			ax.set_ylim(np.min(kwargs['ylim']),np.max(kwargs['ylim']))

	try:
		ax.set_xlabel(kwargs['xlabel'],fontsize=16)

	except KeyError:

		unit = '(GeV/c)' if rescale else '(MeV/c)'

		ax.set_xlabel(r'$p_{\parallel}$ %s'% unit,fontsize=16)

	ax.set_ylabel('Events',fontsize=16)

	ax.tick_params(axis='both',which='major',labelsize=16)

	ax.legend(loc='upper right',fontsize=16)

	return fig,ax

#---------------------------------------------------------------------------------------#
#		Plot momentum distribution and fit
#---------------------------------------------------------------------------------------#

def plot_ppar(self,spec,plot_fit,log,rescale,reac,**kwargs):
	'''Plot experimental momentum distribution with fit if available.'''

	#histogram plot
	fig,ax 	= plot_hist(spec,log,rescale,**kwargs)

	#Convert p to GeV/c
	scale 	= 1e-3 if rescale else 1

	ylim 	= ax.get_ylim()

	#plot fit (if exists)
	try:

		#if plotting theory is not wanted, simply raise an error
		if not plot_fit:
			raise NameError

		#check if reacted or unreacted
		if reac:
			fit_range 	= self.fit_range_reac
			fit_func	= self.fit_func_reac
			fit_res		= self.fit_res_reac.x
		else:
			fit_range 	= self.fit_range_unreac
			fit_func	= self.fit_func_unreac
			fit_res		= self.fit_res_unreac.x

		#x_val_plot = np.linspace(spec['x_centers'][0],spec['x_centers'][-1],1000)*scale
		x_val_plot 	= np.linspace(1.05*fit_range[0]-0.05*fit_range[1],
					      1.05*fit_range[1]-0.05*fit_range[0],
					      1000)

		x_val_0		= fmin(lambda x: -fit_func(x,*fit_res),0,disp=0)
		#x_val_shift	= x_val_plot - fit_res[-2]
		x_val_shift	= x_val_plot - x_val_0
		y_val_plot	= fit_func(x_val_plot,*fit_res)

		ax.plot(x_val_plot*scale,
			y_val_plot,
			linestyle='-',
			color='orange',
			label='Fit spectrum'
			)
		
		ax.plot(x_val_shift*scale,
			y_val_plot,
			linestyle='--',
			color='orange',
			label='Fit shifted'
			)

		#fit range
		ax.axvspan(fit_range[0]*scale,fit_range[1]*scale,
				alpha=0.25,color='grey')

	except (NameError,AttributeError) as error:
		pass

	#change style of y ticks to scientific
	ax.ticklabel_format(axis='y',style='sci',scilimits=(0,2))
	ax.yaxis.get_offset_text().set_visible(False)

	fig.canvas.draw()

	exponent = ax.yaxis.get_major_formatter().orderOfMagnitude
	
	if exponent > 1:
		ax.set_ylabel(r'Events ($10^{%i}$)'% exponent, fontsize=16)
	else:
		ax.set_ylabel(r'Events', fontsize=16)

	#set ylim in case fit goes wrong
	#make sure the y axis ends with a tick
	#can be overwritten by kwargs
	try:
		ax.set_ylim(np.min(kwargs['ylim']),np.max(kwargs['ylim']))
	
	except KeyError:
		
		ylim_scale = 10**exponent*np.ceil((ylim[1]/10**exponent))
		ax.set_ylim(ylim[0],ylim_scale)

	#x label has to be treated here to distinguish between NSCL and RIBF data
	unit = '(GeV/c)' if rescale else '(MeV/c)'

	if self.__class__.__name__ == 'ppar_NSCL':
		ax.set_xlabel(r'$p_{\parallel}-p_{\parallel,0}$ %s'% unit,fontsize=16)
	else:
		ax.set_xlabel(r'$p_{\parallel}$ %s'% unit,fontsize=16)

	#legend only when plot labels are present
	if ax.get_legend_handles_labels() != ([],[]):
		plt.legend(loc='upper right',fontsize=16)
	
	try:
		plt.savefig(kwargs['save'])

	except KeyError:
		pass

	return fig,ax

#---------------------------------------------------------------------------------------#
#		Plot stopping
#---------------------------------------------------------------------------------------#

def plot_stop(data,interpol,beam,product,**kwargs):
	'''Plot correlation between TKE and range from Atima.'''

	fig,ax = plt.subplots(figsize=(10,4))
	fig.subplots_adjust(bottom=0.2)

	for key in data.keys():

		nucl = beam if key=='beam' else product

		ax.plot(data[key][:,1]*1e-3,data[key][:,2]*1e-3,
			marker='x',linestyle='',color=Dic_colors[key])

		x_val = np.linspace(np.min(data[key][:,1]),np.max(data[key][:,1]),1000)

		ax.plot(x_val*1e-3,
			interpol[key]['TKE-range'](x_val)*1e-3,
			color=Dic_colors[key],
			linestyle='--',
			zorder=2,
			label=f'Interpolation {name_nucl(nucl['name'])}',
			)

	#ax.set_xlabel('TKE (MeV)',fontsize=16)
	#ax.set_ylabel(r'Range ($\mu$m)',fontsize=16)
	ax.set_xlabel('TKE (GeV)',fontsize=16)
	ax.set_ylabel(r'Range (mm)',fontsize=16)

	ax.tick_params(axis='both',which='major',labelsize=16)

	ax.legend(fontsize=16)

	try:
		plt.savefig(kwargs['save'])

	except KeyError:
		pass

	return fig,ax

#---------------------------------------------------------------------------------------#
#		Plot theoretical momentum distribution
#---------------------------------------------------------------------------------------#

def plot_theo(ppar_theo,rescale,**kwargs):
	'''Plot theoretical momentum distributions with convolutions if available.'''

	#Convert p to GeV/c
	scale = 1e-3 if rescale else 1

	fig,ax = plt.subplots(figsize=(10,4))
	fig.subplots_adjust(bottom=0.2)

	if 'tail_target' in ppar_theo.keys():

		ax.plot(ppar_theo['orig'].centers*scale,
			#data['orig'].centers+kinematics['after']['p'])*scale,
			ppar_theo['orig'].values/np.sum(ppar_theo['orig'].values),
			linestyle='-',
			color='grey',
			alpha=0.5,
			label='Theory shifted'
			)

		ax.plot(ppar_theo['tail'].centers*scale,
			ppar_theo['tail'].values/np.sum(ppar_theo['tail'].values),
			linestyle='--',
			color='grey',
			alpha=0.5,
			label='w/ unreacted'
			)

		ax.plot(ppar_theo['tail_target'].centers*scale,
			ppar_theo['tail_target'].values/np.sum(ppar_theo['tail_target'].values),
			linestyle='-',
			color='black',
			label='w/ unreacted \nand target'
			)

		x_label = r'$p_{\parallel}$'

	else:

		ax.plot(ppar_theo['orig'].centers*scale,
			ppar_theo['orig'].values/np.sum(ppar_theo['orig'].values),
			linestyle='-',
			color='black',
			label='Theory rest frame'
			)

		x_label = r'$p_{\parallel}$'
		#x_label = r'$p_{\parallel}-p_{\parallel,0}$'

	if rescale:
		ax.set_xlabel(x_label+' (GeV/c)',fontsize=16)
	else:
		ax.set_xlabel(x_label+' (MeV/c)',fontsize=16)

	ax.set_ylabel('Density',fontsize=16)

	ax.tick_params(axis='both',which='major',labelsize=16)

	ax.legend(fontsize=16)

	try:
		plt.savefig(kwargs['save'])

	except KeyError:
		pass

	return fig,ax

#---------------------------------------------------------------------------------------#
#		Plot experimental momentum distribution with theory
#---------------------------------------------------------------------------------------#

def plot_mode(self,spec,plot_theory,log,rescale,show_region,**kwargs):
	'''Plot experimental exclusive momentum distributions with fit if available.'''

	#Convert p to GeV/c
	x_scale = 1e-3 if rescale else 1
	label 	= kwargs['label'] if 'label' in kwargs else ['Experiment','Theory']

	fig,ax = plt.subplots(figsize=(10,4))
	fig.subplots_adjust(bottom=0.2)

	#old plot appareance
	# for num_ind,ind in enumerate(spec['ind_nonzero']):

		# diff = np.diff(spec['x_centers'])[0]

		# ax.plot([spec['x_centers'][ind]-0.5*diff,spec['x_centers'][ind]+0.5*diff],
		# 	[spec['mode'][ind],spec['mode'][ind]],
		# 	color='black'
		# 	)

		# ax.fill_between([spec['x_centers'][ind]-0.5*diff,spec['x_centers'][ind]+0.5*diff],
		# 	spec['sc_interval'][ind][0],spec['sc_interval'][ind][1],
		# 	color='grey',alpha=0.5
		# 	)

	ind = spec.ind_nonzero
	err = np.abs(spec.sc_interval[ind].T-spec.mode[ind])

	ax.errorbar(spec.centers[ind]*x_scale,
		    spec.mode[ind],
		    yerr=err,
		    linestyle='',
		    marker='x',
		    capsize=5,
		    color='black',
		    label=label[0],
		   )

	#plot theory (if exists)
	try:

		#if plotting theory is not wanted, simply raise an error
		if not plot_theory:
			raise NameError

		#fit_range 	= self.fit_range

		x_val_plot 	= self.ppar_theo['tail_target'].centers
		y_val_plot 	= self.ppar_theo['tail_target'].interpol(x_val_plot)

		#shift theory horizontally only if fitted
		if len(self.fit_res) > 1:

			x_val_plot = x_val_plot + self.fit_res[1]

		#apply scaling factor for agreement with experimental data
		y_val_plot 	= y_val_plot*self.fit_res[0]#*self.params['rebin']

		ax.plot(x_val_plot*x_scale,
			y_val_plot,
			linestyle='-',
			color='orange',
			label=label[1]
			)

		#fit range if requested and fitted
		if show_region and len(self.fit_res) > 1:

			ax.axvspan(self.fit_range[0]*x_scale,self.fit_range[1]*x_scale,
					alpha=0.25,color='grey')

	except (NameError,AttributeError) as error:
		pass

	if log:
		ax.set_yscale('log')

	#change style of y ticks to scientific
	ax.ticklabel_format(axis='y',style='sci',scilimits=(0,2))
	ax.yaxis.get_offset_text().set_visible(False)

	fig.canvas.draw()

	exponent = ax.yaxis.get_major_formatter().orderOfMagnitude

	if exponent > 1:
		ax.set_ylabel(r'Events ($10^{%i}$)'% exponent, fontsize=16)
	else:
		ax.set_ylabel(r'Events', fontsize=16)

	ax.tick_params(axis='both',which='major',labelsize=16)

	#set y limits, can be overwritten with kwargs
	ylim = ax.get_ylim()
	ylim_scale = 10**exponent*np.ceil((ylim[1]/10**exponent))
	ax.set_ylim(0,ylim_scale)

	#plot limits from kwargs
	for key in kwargs:

		if key == 'xlim':
			ax.set_xlim(np.min(kwargs['xlim'])*x_scale,np.max(kwargs['xlim'])*x_scale)
		elif key == 'ylim':
			ax.set_ylim(np.min(kwargs['ylim']),np.max(kwargs['ylim']))

	#possibility to change the x label
	try:
		ax.set_xlabel(kwargs['xlabel'],fontsize=16)

	except KeyError:

		if rescale:
			ax.set_xlabel(r'$p_{\parallel}$ (GeV/c)',fontsize=16)
		else:
			ax.set_xlabel(r'$p_{\parallel}$ (MeV/c)',fontsize=16)

	#if 'label' in kwargs:

	ax.legend(fontsize=16)

	try:
		plt.savefig(kwargs['save'])

	except KeyError:
		pass

	return fig,ax
