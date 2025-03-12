'''Operations on 1d spectra and fit'''

import re
import uproot

import numpy as np

from copy import deepcopy
from scipy.optimize import minimize
#from scipy.stats import norm

from .utils import gaussian,left,right,piecewise,mode,sc_interval

#---------------------------------------------------------------------------------------#
#		Load data and isolate histogram data
#---------------------------------------------------------------------------------------#

def load_file(file_name,hist_name,**kwargs):
	'''Load .root files and extract histogram data.'''

	with uproot.open(file_name) as file:

		try:
			hist_name 	= re.findall(hist_name,'|'.join(file.classnames().keys()))[0]

		except IndexError:

			hist_names 	= [hist for hist in file.classnames().keys() if 'GVALUE' not in hist]

			raise FileNotFoundError(f'histogram name {hist_name} not found \
						in {hist_names}!')


		xaxis		= file[hist_name].all_members['fXaxis']
		values		= file[hist_name].values()

		spec		= {'x_edges':	xaxis.edges(),
				   'x_centers':	xaxis.centers(),
				   'values':	values,
				  }
		
		try:

			spec['mc_values']	= np.array([np.random.poisson(i,int(kwargs['mc_num'])) for i in values])

		except KeyError:

			pass
		
		return spec

def prepare_spec(file_name,hist_name,kinematics,beam,product):
	'''Obtain and prepare histogram data.'''

	#Load data
	data 			= load_file(file_name,hist_name)

	#Account for nucleons lost in reaction
	corr_fac		= product['ion mass']/beam['ion mass']

	data['x_centers']      *= corr_fac
	data['x_edges']	       *= corr_fac

	return data

def shift_spec(spec,params):

	spec_shifted = deepcopy(spec)

	spec_shifted['x_centers'] -= params.x[-2]
	spec_shifted['x_edges']   -= params.x[-2]

	return spec_shifted

#---------------------------------------------------------------------------------------#
#		Rebin histogram
#---------------------------------------------------------------------------------------#

def rebin_spec(spec,rebin):
	'''Rebin histogram.'''

	if rebin == 1:

		return spec

	bins 	= len(spec['x_centers'])

	if bins % rebin:

		raise ValueError(f'number of bins ({bins}) must be an integer multiple \n \
				of the rebin factor {rebin}!')

	#values
	if spec['values'].ndim == 1:

		values 	= np.sum(spec['values'].reshape(int(bins/rebin),rebin),axis=1)

	elif spec['values'].ndim == 2:

		mc_num 	= spec['values'].shape[1]
		values 	= np.sum(spec['values'].reshape(int(bins/rebin),rebin,mc_num),axis=1)

	else:
		raise ValueError(f'dimension of values ({spec["values"].ndim}) must be one or two!')

	#edges and centers
	if rebin % 2:

		#edges 	= spec['x_edges'][::rebin]
		centers = spec['x_centers'][int(rebin/2)::rebin]

	else:
		#edges 	= spec['x_edges'][::rebin]
		centers = spec['x_edges'][int(rebin/2)::rebin]

	return {#'x_edges':	edges,
		'x_centers':	centers,
		'values':	values,
		}

#---------------------------------------------------------------------------------------#
#		Evaluate histogram
#---------------------------------------------------------------------------------------#

def eval_spec(spec,threshold):
	'''Calculate mode and shortest coverage interval for each bin.'''

	len_spec 		= len(spec['x_centers'])

	spec['ind_nonzero'] 	= []
	spec['mode']		= np.zeros(len_spec)
	spec['sc_interval'] 	= np.zeros((len_spec,2))

	#values = spec['mc_values'] if 'mc_values' in spec.keys() else spec['values']
	#mc_num = values.shape[1]

	#for i,j in enumerate(values):
	for i,j in enumerate(spec['values']):

		if not np.all(j==0) and np.mean(j) > threshold:
		#if np.count_nonzero(j) > np.sqrt(mc_num) and np.mean(j) > threshold:

			spec['ind_nonzero'].append(i)

			spec['mode'][i]        = mode(j)
			spec['sc_interval'][i] = sc_interval(j)

	return spec

#---------------------------------------------------------------------------------------#
#		Fit histogram
#---------------------------------------------------------------------------------------#

def piecewise_minimize(params,x,y,gauss):
	'''Fit residuals of piecewise function with penalty term for continuity.'''

	#args[0] = switch

	#left
	#args[1] = exp1
	#args[2] = exp2
	#args[3] = area
	#args[4] = mean
	#args[5] = sigma

	#right
	#args[6] = height
	#args[7] = width
	#args[8] = center
	#args[9] = sigma

	#add a shift in index
	shift = 3 if gauss else 0

	fit 		= piecewise(x,gauss,params)

	residuals 	= np.linalg.norm(y-fit)
	penalty   	= np.linalg.norm(left(params[0],gauss,*params[1:3+shift])-\
				   	 right(params[0],*params[3+shift:]))

	return residuals + penalty

def right_minimize(params,x,y):
	'''Fit residuals of asymmetric peak.'''

	fit		= right(x,*params)
	residuals 	= np.linalg.norm(y-fit)

	return residuals

def gaussian_minimize(params,x,y):
	'''Fit residuals of Gaussian peak.'''

	fit		= gaussian(x,*params)
	residuals 	= np.linalg.norm(y-fit)

	return residuals

#def fit_spec(spec,fit_range,x0,gauss):
def fit_spec(spec,fit_range,x0):
	'''Fit of experimental momentum distributions with low-momentum tail.'''

	mask  	= (spec['x_centers'] >= fit_range[0])*\
		  (spec['x_centers'] <= fit_range[1])

	x_val 	= spec['x_centers'][mask]
	y_val 	= spec['values'][mask]

	#opt_res_min 	= minimize(piecewise_minimize,
	#			args=(x_val,y_val,gauss),
	#			x0=x0,
	#			method='Nelder-Mead',
	#			options={'maxiter':10000})

	if len(x0) == 3:
		opt_res_min 	= minimize(gaussian_minimize,
						args=(x_val,y_val),
						x0=x0,
						method='Nelder-Mead',
						options={'maxiter':10000})
	elif len(x0) == 4:
		opt_res_min 	= minimize(right_minimize,
						args=(x_val,y_val),
						x0=x0,
						method='Nelder-Mead',
						options={'maxiter':10000})
	elif len(x0) == 7:
		opt_res_min 	= minimize(piecewise_minimize,
						args=(x_val,y_val,False),
						x0=x0,
						method='Nelder-Mead',
						options={'maxiter':10000})
	else:
		opt_res_min 	= minimize(piecewise_minimize,
						args=(x_val,y_val,True),
						x0=x0,
						method='Nelder-Mead',
						options={'maxiter':10000})

	return opt_res_min

#def fit_peak(spec,fit_range,x0):
	'''Fit of experimental momentum distributions without low-momentum tail.'''

#	mask  	= (spec['x_centers'] >= fit_range[0])*\
#		  (spec['x_centers'] <= fit_range[1])

#	x_val 	= spec['x_centers'][mask]
#	y_val 	= spec['values'][mask]

#	opt_res_min 	= minimize(right_minimize,
#				args=(x_val,y_val),
#				x0=x0,
#				method='Nelder-Mead',
#				options={'maxiter':10000})

#	return opt_res_min
