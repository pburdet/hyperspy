
s = utils_eds.database_3Dspec('SEM')

#basic pca
s.change_dtype('float')
s.decomposition(True)
s.plot_explained_variance_ratio()
s.plot_decomposition_results()

sr = s.get_decomposition_model(5)

s.blind_source_separation(5)
s.plot_bss_results()

#extra spectrum from standard
s = utils_eds.database_3Dspec('SEM')
from hyperspy.misc.config_dir import config_path
s.add_elements(['Hf','Ta'])
s.link_standard(config_path+'/database/std_RR')
s = s.add_standards_to_signal('all')

s.change_dtype('float')
s.decomposition(True)
sr = sr.get_decomposition_model(5)
sr[102:134,125:152].get_lines_intensity(
     plot_result=True,lines_deconvolution='standard')
     
#back fitting    
s = utils_eds.database_4Dspec('TEM')[::,::,1]
s.change_dtype('float')
dim = s.axes_manager.shape
s = s.rebin((dim[0]/4, dim[1]/4, dim[2]))
s2 = s.deepcopy()
s2 = s2.rebin((dim[0]/8, dim[1]/8, dim[2]))
s2.decomposition(True)
rs = s.get_decomposition_model_from(s2, components=5)
