"""Example of EDS spec from scratch

"""

test = True
# loading the rpl file
if test :
    s = utils_eds.database_3Dspec('noisy').as_spectrum(0)
    s.set_signal_type('EDS_SEM')
else :
    s = load('your_file.rpl',record_by="spectrum",signal_type="EDS_SEM")
    
# Energy axis calibration from 1 pix msa
if test :
    s_1pix = utils_eds.database_1Dspec('noisy')
    s_1pix.set_signal_type('EDS_SEM')
else : 
    s_1pix = = load('your_file.msa',signal_type="EDS_SEM")
s.get_calibration_from(s_1pix)
s.axes_manager[-1].name = 'E'

# Spatial axes calibration
axes_name = ['x','y']
units_name = '${\mu}m$'
scale = array([0.04,0.04])
for i in range(2):
    s.axes_manager[i].name = axes_name[i]
    s.axes_manager[i].units = units_name
    s.axes_manager[i].scale = scale[i]

# Set elements and lines
s.set_elements(['Al','Zn'])
s.add_lines()

# Plotting the lines intensity
s.get_lines_intensity(plot_result=True)
