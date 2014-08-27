"""Example

"""

#SEM BAM
s = database.spec1D()
m = create_model(s,auto_background=False)
m.add_lines(['Ar_Ka','O_Ka'])
m.add_background([1,2,3],detector_name='Xmax')
    # , weight_fraction=[0.25,0,0.25,0.25,0.25])
m.fit()
m.fit_background(start_energy=0.0)
m.fit_energy_resolution()
m.fit_sub_xray_lines_weight(['Cu_La','Mn_La','Zr_La'],bound=1)
m.fit_xray_lines_energy(bound=0.02)
m.fit()
           
#TEM
s = database.spec1D('TEM')
m = create_model(s,auto_background=False)
m.add_family_lines(['Cu_Ka','Cu_La'])
m.add_background(detector_name='osiris' 
                 ,thicknesses = [75,100,125])
m.fit()
m.fit_background()
m.fit_xray_lines_energy(['Cr_Ka'],bound=20)
m.fit_energy_resolution()
m.fit_sub_xray_lines_weight(bound=1)#Might be slow

#Determining the scaling (fast method)
m = create_model(s,auto_add_lines=False,auto_background=False)
line_to_fit = 'Ni_Ka'
m.add_family_lines([line_to_fit])
m.fit()
m.fit_xray_lines_energy([line_to_fit],bound=20)
scale = s.axes_manager[-1].scale
energy_fit = m[line_to_fit].centre.value 
energy_line = s._get_line_energy(line_to_fit)
s.axes_manager[-1].scale = scale - (
        energy_fit-energy_line) / energy_line * scale

#Plotting
m.plot(plot_components=True)
m.plot()
utils.plot.plot_spectra([s,m.as_signal(),s-m.as_signal()],
                        legend=['spectrum','model','residual'])
for bc in m.background_components:
    print bc.name
    print bc.yscale.value
    
from hyperspy.drawing.utils import animate_legend
m.plot(plot_components=True)
legend(['spectrum','model']+[co.name for co in m])
animate_legend()
    
#TEM quant
m.get_lines_intensity(xray_lines='from_metadata')
s.get_kfactors_from_brucker()
s.quant_cliff_lorimer()
weight_fraction = utils.stack(s.metadata.Sample.quant).data
