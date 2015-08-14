"""Example

Importing an EDS spectral image, calibrate it and plot X-ray line intensity.
This example is done for file exported from INCA, oxford instrument.
It might need to be change for other

"""
%hyperspy -r qt4


your_own_file = False
# loading the rpl file and a spectrum from one pixel
if your_own_file:
    s = hs.load('your_file.rpl', signal_type="EDS_SEM").as_spectrum(0)
    s_1_pixel = hs.load('your_file_1_pixel.msa', signal_type="EDS_SEM")
else:
    others = 'database'
    #others = 'model'
    if others == 'database':
        s = database.spec3D('noisy').as_spectrum(0)
        s.set_signal_type('EDS_SEM')
        s_1_pixel = database.spec1D('noisy')
        s_1_pixel.set_signal_type('EDS_SEM')
    elif others == 'model':
        s = utils_eds.simulate_model(['Al', 'Zn'], shape_spectrum=[2, 3, 1024])
        s_1_pixel = utils_eds.simulate_model(['Al', 'Zn'])
    elif others == 'model2':
        s = hs.signals.EDSSEMSpectrum(range(1024))
        s.axes_manager[-1].scale = 0.01
        s.axes_manager[-1].units = "keV"
        s.axes_manager[-1].offset = -0.1
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.counts_rate = 10
        s.set_microscope_parameters(beam_energy=15, live_time=10)
        s.set_elements(['Al', 'Zn'])
        s = hs.stack([s.simulate_model()+0.5 for i in range(3)])
        s = hs.signals.EDSTEMSpectrum([list(s.data)*10]*10)
        s.add_poissonian_noise()
    else:
        # Build a spectrum
        data = range(1024 / 2) + range(1024 / 2, 0, -1)
        s_1_pixel = signals.EDSSEMSpectrum(data)

        s_1_pixel.axes_manager[-1].scale = 0.01
        s_1_pixel.axes_manager[-1].units = "keV"
        s_1_pixel.axes_manager[-1].offset = -0.1
        s_1_pixel.set_microscope_parameters(beam_energy=15, live_time=10)

        # Build a map
        data = [data] + [data[::-1]] + [list(sqrt(data))]
        data = [data] + [list(power(data[::-1], 2))]
        s = signals.EDSSEMSpectrum(data)

# Energy axis calibration contains in s_1_pixel is tranfer to s.

s.get_calibration_from(s_1_pixel)
s.axes_manager[-1].name = 'E'


# Spatial axes calibration
axes_name = ['x', 'y']
units_name = '${\mu}m$'
scale = np.array([0.04, 0.04])
for i in range(2):
    s.axes_manager[i].name = axes_name[i]
    s.axes_manager[i].units = units_name
    s.axes_manager[i].scale = scale[i]

# Set elements and lines
s.set_elements(['Al', 'Zn'])
s.add_lines()

# Plotting the lines intensity
plt.set_cmap('RdYlBu_r')
s.get_lines_intensity(plot_result=True)
