'''Screen output for ppar'''

#---------------------------------------------------------------------------------------#
#		Header message
#---------------------------------------------------------------------------------------#

def header_message(text):
	'''Header message for sections.'''

	print('########################')
	print(f'#  {text:<15}     #')
	print('########################')
	print()

#---------------------------------------------------------------------------------------#
#		Start message
#---------------------------------------------------------------------------------------#

def start_message(text):
	'''Start message.'''

	print('##############################')
	#print('#                           #')
	print('#       ppar convolver       #')
	print(f'#            {text}            #')
	#print('#                           #')
	print('##############################')
	print()

#---------------------------------------------------------------------------------------#
#		Overview beam/target/product
#---------------------------------------------------------------------------------------#

def nucl_message(beam,target,product):
	'''Message with input for beam, target, and product.'''

	header_message('Input')

	print(f'{"":<20}{"Beam":<15}{"Target":<15}{"Product":<15}')
	print('---------------------------------------------------------')

	for key in ['name','A','Z']:

		#if key == 'A':

		#	print(f'{key:<20}{beam[key]:<15}{"":<15}{product[key]:<15}')

		#else:
		print(f'{key:<20}{beam[key]:<15}{target[key]:<15}{product[key]:<15}')

	for key in ['mass','ion mass']:

		print(f'{key+" (u)":<20}{beam[key]:<15.4f}{target[key]:<15.4f}{product[key]:<15.4f}')

	print(f'{"density (g/cm3)":<20}{"":<15}{target["density"]:<15.4f}{"":<15}')
	print(f'{"d (mg/cm2)":<20}{"":<15}{target["thickness"]:<15.1f}{"":<15}')
	print(f'{"d (µm)":<20}{"":<15}{target["d"]:<15.1f}{"":<15}')

	print()

#---------------------------------------------------------------------------------------#
#		Kinematics
#---------------------------------------------------------------------------------------#

def kinematics_message(kinematics,beam,target,product):
	'''Message with details on S800 settings.'''

	print(f'Reaction:\t{target["name"]}({beam["name"]},{product["name"]}){target["name"]}')
	print()
	print(f'\t{"":<20}{"Seg7/BTS33":<15}{"Seg8/BTS34":<15}')
	print('\t---------------------------------------------')

	print(f'\t{"Brho (Tm)":<20}{kinematics["before"]["Brho"]:<15.4f}{kinematics["after"]["Brho"]:<15.4f}')
	print(f'\t{"Nucleus":<20}{kinematics["before"]["nucl"]:<15}{kinematics["after"]["nucl"]:<15}')
	print(f'\t{"TKE (MeV)":<20}{kinematics["before"]["TKE"]:<15.2f}{kinematics["after"]["TKE"]:<15.2f}')
	print(f'\t{"p (MeV/c)":<20}{kinematics["before"]["p"]:<15.2f}{kinematics["after"]["p"]:<15.2f}')
	print(f'\t{"gamma ( )":<20}{kinematics["before"]["gamma"]:<15.4f}{kinematics["after"]["gamma"]:<15.4f}')
	print(f'\t{"beta ( )":<20}{kinematics["before"]["beta"]:<15.4f}{kinematics["after"]["beta"]:<15.4f}')

	print()

#---------------------------------------------------------------------------------------#
#		Stopping
#---------------------------------------------------------------------------------------#

def stopping_message(method,data,calls,threshold,nucl,target):
	'''Message with details on stopping calculations.'''

	print(f'Stopping of {nucl["name"]} in {target["thickness"]:.1f} mg/cm2 {target["name"]}: ')
	print(f'\tFound solution within {threshold:.1f} µm after {calls}({calls+2}) iterations(Atima calls).')
	#print()

	#if method == 'fw':

	#	print(f'\tafter target:\tE   = {data[-1,0]:>8.2f} MeV/u')
	#	print(f'\t\t\tTKE = {data[-1,1]:>8.2f} MeV')

	#else:
	#	print(f'\tbefore target:\tE   = {data[-1,0]:>8.2f} MeV/u')
	#	print(f'\t\t\tTKE = {data[-1,1]:>8.2f} MeV')

	#out_string  = f'Found solution within {threshold} µm for stopping of {nucl["name"]} '
	#out_string += f'in {target["name"]} after {calls}({calls+2}) iterations(Atima calls).'

	#print(out_string)
	print()
