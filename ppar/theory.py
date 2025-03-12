'''Treatment of theoretical momentum distributions'''

import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import minimize

from .spectrum import piecewise
from .utils import gaussian,right

#---------------------------------------------------------------------------------------#
#		Load theoretical momentum distribution
#---------------------------------------------------------------------------------------#

def load_theo(file_name):
	'''Load theoretical momentum distribution from file.'''

	theo = np.loadtxt(file_name)

	data = {}

	data['orig'] = 	{'x_centers':	theo[:,0],
			 'values':	theo[:,1],
			 'interpol':	interp1d(theo[:,0],theo[:,1],
			   			kind='cubic'),
			}

	return data

#def prepare_theo(file_name,kinematics):
	'''Load theoretical momentum distribution.'''

#	data = {}

	#Load data
#	theo 		= load_theo(file_name)

#	data['orig'] 	= {'x_centers':	theo['x_centers'],#*1e-3,
#			   'values':	theo['values'],
#			   'interpol':	interp1d(theo['x_centers'],theo['values'],
#			   			kind='cubic'),
#			   }

	#Move it to the correct momentum
	#x_centers 	= theo['x_centers'] + kinematics['after']['p']

	#Boost it to the lab frame
	#x_centers      *= kinematics['after']['gamma']

	#Move it back
	#x_centers      -= kinematics['after']['p']

	#Convert momentum to GeV/c
	#x_centers      *= 1e-3

	#data['boost'] 	= {'x_centers':	x_centers,
	#		   'values':	theo['values'],
	#		   'interpol':	interp1d(x_centers,theo['values'],
	#		   			kind='cubic'),
	#		   }

#	return data

#---------------------------------------------------------------------------------------#
#		Convolve theoretical momentum distribution
#---------------------------------------------------------------------------------------#

def filter(x,limits):
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

	p				= ppar_theo['orig']['x_centers']
	
	if boost:
		p *= kwargs['kinematics']['after']['gamma']
	
	p_sized 			= np.linspace(2*p[0],2*p[-1],2*len(p)-1)

	values 				= ppar_theo['orig']['values']

	#include effect of reaction position
	#p_filter 			= filter(p,p_range-kinematics['after']['p'])
	p_filter 			= filter(p_sized,p_range)

	values_target 			= np.convolve(values,p_filter,mode='valid')

	ppar_theo['target'] 		= {'x_centers':	p,#+kinematics['after']['p'],
					   'values':	values_target,
					   'interpol':	interp1d(p,values_target,
					   			kind='cubic'),
					   }

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

	#try:
	#	val_unreacted 		= kwargs['fit_val']
	#except KeyError:
		
	#	if len(fit_res.x) == 4:
	#		val_unreacted	= right(p_sized+fit_res.x[-2],*fit_res.x)
	#	else:
	#		val_unreacted 	= piecewise(p_sized+fit_res.x[-2],params['gauss'],fit_res.x)

	values_tail 			= np.convolve(values,val_unreacted,mode='valid')

	ppar_theo['tail'] 		= {'x_centers':	p,#+kinematics['after']['p'],
					   'values':	values_tail,
					   'interpol':	interp1d(p,values_tail,
					   			kind='cubic'),
					   }
 
	#include shape of unreacted momentum distribution (p_par-p_par,0)
	#and effect of reaction position
	values_tail_target 		= np.convolve(values_tail,p_filter,mode='valid')

	ppar_theo['tail_target'] 	= {'x_centers':	p,#+kinematics['after']['p'],
					   'values':	values_tail_target,
					   'interpol':	interp1d(p,values_tail_target,
					   			kind='cubic',
					   			bounds_error=False,
					   			fill_value=(0,0)),
					   }

	return ppar_theo

#---------------------------------------------------------------------------------------#
#		Fit convoluted theoretical momentum distribution to data
#---------------------------------------------------------------------------------------#

def shift_scale_theo(params,x,y,scale,ppar_theo):
	'''Fit function for horizontal shift of convolved momentum distributions.'''

	x_shift 	= x - params[0]
	fit 		= scale*ppar_theo['interpol'](x_shift)

	#x_shift 	= x - params[1]
	#fit 		= params[0]*ppar_theo(x_shift)

	residuals 	= np.linalg.norm(y-fit)

	return residuals

def scale_theo(spec,ppar_theo):
	'''Scale convolved momentum distributions
	vertically using the maximum bin content.'''

	#for y take the modes of the MC values
	x 	= spec['x_centers']
	y 	= spec['mode']

	#position of maximum y value for scaling
	x_max 	= x[np.argmax(y)]
	y_max 	= np.max(y)

	#scale theory to experiment
	scale 	= y_max/np.max(ppar_theo['values'])

	return np.array([scale])

def fit_theo(spec,ppar_theo,fit_range):
	'''Scale convolved momentum distributions and fit them horizontally.'''

	scale 	= scale_theo(spec,ppar_theo)[0]

	#for y take the modes of the MC values
	x 	= spec['x_centers']
	y 	= spec['mode']

	#position of maximum y value for scaling
	x_max 	= x[np.argmax(y)]
	#y_max 	= np.max(y)

	#scale theory to experiment
	#scale 	= y_max/np.max(ppar_theo['values'])

	mask  	= (x >= fit_range[0])*(x <= fit_range[1])

	#cut to range
	x_val 	= spec['x_centers'][mask]
	y_val 	= y[mask]

	res_fit_min = minimize(shift_scale_theo,
				args=(x_val,y_val,scale,ppar_theo),
				x0=x_max,
				options={'maxiter':10000})

	#res_fit_min.x = np.array([scale,res_fit_min.x[0]])

	return np.array([scale,res_fit_min.x[0]])
