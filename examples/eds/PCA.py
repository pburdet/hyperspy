"""
PCA
"""
pyplot.set_cmap('RdYlBu_r')

elements = ["C", "Al", "Ti", "Cr", "Co", "Ni", "Mo", "Hf", "Ta", 'Zr']
s = database.spec3D('SEM')
s.add_elements(elements)

# basic pca
s.change_dtype('float')
s.decomposition(True)
s.plot_explained_variance_ratio()
s.plot_decomposition_results()

f = s.get_decomposition_factors()
f.get_calibration_from(s)
f.add_elements(s.metadata.Sample.elements)
f.plot(True, navigator='slider')

sr = s.get_decomposition_model(5)

s.blind_source_separation(5)
s.plot_bss_results()

sj = s.deepcopy()
sj.data = sj.data + 1j*sr.data
sj.plot()

#PCA for TEM (masking vacum)

s = database.spec3D('TEM')
# dim = s.axes_manager.shape
# s = s.rebin([dim[0],dim[1],2000])

mask = (s.sum(-1) < 28) 
mask.plot()
s.change_dtype('float')
s.decomposition(True,navigation_mask=mask.data)
s.learning_results.loadings = np.nan_to_num(s.learning_results.loadings)
sr = s.get_decomposition_model(3)

s.plot_explained_variance_ratio()
s.plot_decomposition_results()
s.blind_source_separation(3)

loa = s.get_bss_loadings()
loa.plot()

# PCA saving memory

s.change_dtype('float32')
s.decomposition(True)
s.learning_results.crop_decomposition_dimension(30)
s.change_dtype('int16')
# do not try to save the model, just the s with the matrix score

# extra spectrum from standard
s = database.spec3D('SEM')
from hyperspy.misc.config_dir import config_path
s.add_elements(['Hf', 'Ta'])
s.link_standard(config_path + '/database/SEM/std_RR')
s = s.add_standards_to_signal('all')

s.change_dtype('float')
s.decomposition(True)
sr = s.get_decomposition_model(5)
sr[102:134, 125:152].get_lines_intensity_old(
    plot_result=True, lines_deconvolution='standard')

# back fitting
s = database.spec4D('TEM')[64:, 64:, 1]
s.set_microscope_parameters(beam_energy=200)
s.change_dtype('float')
dim = s.axes_manager.shape
s = s.rebin((dim[0] / 4, dim[1] / 4, dim[2]))
s2 = s.deepcopy()
s2 = s2.rebin((dim[0] / 8, dim[1] / 8, dim[2]))
s2.decomposition(True)
rs = s.get_decomposition_model_from(s2, components=5)
