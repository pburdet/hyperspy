"""
"""

elements = ['Al', 'Zn']

# Model

# 1D spec BAM
s1 = utils_eds.simulate_model()
s1.metadata.General.title = 'model BAM'

# elements
s2 = utils_eds.simulate_model(elements)
s2.metadata.General.title = 'model el'

# all param

s3 = utils_eds.simulate_model(elements,
                              beam_energy=10,
                              live_time=30,
                              weight_percents=[0.1, 0.9],
                              energy_resolution_MnKa=128,
                              counts_rate=50000)
s3.metadata.General.title = 'model param'

# map
s = utils_eds.simulate_model(elements, shape_spectrum=[2, 3, 1024])
s.plot()


# Monte Carlo

sm = utils_eds.database_1Dspec()
s4 = utils_eds.simulate_one_spectrum(100, mp=sm.metadata,
                                     dose=1,
                                     compo_at=[0.6, 0.4],
                                     elements=elements)
s4.metadata.General.title = 'Monte Carlo'

utils.plot.plot_spectra([s1, s2, s3, s4], legend='auto')
