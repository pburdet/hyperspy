"""
Quantification
"""

plt.set_cmap('RdYlBu_r')

long_time = False
elements = ['Ti', 'Fe', 'Ni']
xray = ['Ti_Ka', 'Fe_Ka', 'Ni_La']
#elements = ['Ti', 'Fe', 'Ni']
#xray = ['Ti_Ka', 'Fe_Ka', 'Ni_La']
from hyperspy.misc.config_dir import config_path

s3 = database.spec3D('Ti_SEM')
s3.set_microscope_parameters(live_time=0.12)
s3.set_elements(elements)
s3.set_lines(xray)
s3.link_standard(config_path + '/database/SEM/std_TiFeNi')


s4 = database.spec4D()
r4 = database.result3D()
s4.get_calibration_from(r4)
s4.link_standard(config_path + '/database/SEM/std_RR')


s3.get_take_off_angle()
s4.get_take_off_angle()

""" Standard quantification
"""

# 1D spec

s1 = s3[100, 50]
s1.get_kratio()
s1.check_kratio(xray, top_hat_applied=True, plot_all_standard=True)
s1.quant()

# 3D map standard quant

s3.get_kratio()

s3.plot_histogram_result('kratios')
s3.get_result(xray[0], 'kratios')

s3.quant()

s3.plot_histogram_result('quant')
s3.get_result(elements[0], 'quant')

# image_eds.phase_inspector(s.metadata.Sample.quant)

# 4D enhanced standard quant


if long_time is False:
    s4 = s4[50:60, 40:50]

r = s4.get_lines_intensity(plot_result=False)
image_eds.plot_orthoview_animated(r[2])

s4.get_kratio([[['Ni_Ka', 'Co_Ka'], ['Ni', 'Co'], [6.7, 7.75]],
               [["Ta_Ma", "Hf_Ma", 'Al_Ka'], ["Ta", "Hf", 'Al'], [1.25, 1.95]]], plot_result=False)

s4.quant(plot_result=False)

s4.align_results(reference=['kratios', 6], starting_slice=0)

r = s4.get_result(xray[0], 'kratios')
image_eds.plot_orthoview_animated(r)
s4.plot_histogram_result('kratios')

r = s4.get_result(elements[0], 'quant')
image_eds.plot_orthoview_animated(r)
s4.plot_histogram_result('quant')

""" Ehnanced quantification
"""

if long_time:
    a, b = s4.simulate_electron_distribution(nb_traj=100000,
                                             limit_x=[-0.400, 0.700], dx0=0.01, dx_increment=0.75, plot_result=True)
else:
    s4.metadata.elec_distr = r4.metadata.elec_distr

s4.plot_electron_distribution()

#s.mapped_parameters.elec_distr.max_slice_z = [ 6,  6,  6,  6,  5,  5,  6,  5,  6]
#s.quant(enh=True, enh_param=[0, 0.001, 0.005, 49, 1], compiler=0)
# for TiFeNi
# a,b=s.simulate_electron_distribution(nb_traj=100000,
#    limit_x=[-0.35,0.45], dx0=0.008, dx_increment = 0.5, plot_result=True)
# for AlZn
# a,b=s.simulate_electron_distribution(nb_traj=100000,
#    limit_x=[-0.25,0.3], dx0=0.004, dx_increment = 0.75, plot_result=True

###
if 1 == 0:
    s4.metadata.elec_distr.max_slice_z = [6, 6, 6, 6, 5, 5, 6, 5, 6, 6]
    s4.quant(plot_result=False, enh=True,
             enh_param=[0, 0.001, 0.005, 49, 1], compiler=0)
    err = s4.read_enh_ouput(compiler=0)

    r = s4.get_result(elements[0], 'quant_enh')
    image_eds.plot_orthoview_animated(r)
    s4.plot_histogram_result('quant_enh')
    err[::, 2].get_histogram().plot()
