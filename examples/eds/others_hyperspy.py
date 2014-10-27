
#Gui to calibrate
s.calibrate()

#Find a peak, could be used to auto recogintion of line
s.find_peaks1D_ohaver()

#Need to be used for two windows method
s = database.spec1D('TEM')
b = s.deepcopy()
b.interpolate_in_between(5150.,5680.,delta=10,deg=1,show_progressbar=False)
c = s.deepcopy()
c.interpolate_in_between(5200.,5650.,delta=20,kind='spline+',
    show_progressbar=False)
utils.plot.plot_spectra([s,b,s-b,c,s-c],legend=['Spectrum', 'Linear bck',
    'residual','Spline bck', 'residual'])
ylim(-1000,20000)
xlim(5000.,5800.)
fig = gcf()
fig

#Frequency cut off adn smoothing
s.hanning_taper

s.smooth_lowess
s.smooth_savitzky_golay
s.smooth_tv


s.swap_axes
