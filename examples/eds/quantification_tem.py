# Cliff-Lorimer

# simple method

s = database.spec3D('TEM_robert')
s.set_elements(['Al','Cr', "Ni"])
s.set_lines(["Al_Ka", "Cr_Ka", "Ni_Ka"])
kfactors = [s.metadata.Sample.kfactors[0], s.metadata.Sample.kfactors[3],
            s.metadata.Sample.kfactors[7]]
bc = s.estimate_background_windows()
s.sum(0).sum(0).plot(background_windows=bc)
intensities = s.get_lines_intensity(background_windows=bc)
res = s.quantification(intensities, kfactors)
hs.plot.plot_signals(res)

# Simulate two elements standard
s.set_microscope_parameters(live_time=30)
s.simulate_two_elements_standard(nTraj=100)
s.get_kfactors_from_standard()
s.metadata.Sample.intensities = intensities
s.quantification_old()

# kfactors from first principles
s.get_kfactors_from_first_principles()
s.quantification_old()

# Quant of PCA

mask = (s.sum(-1) > 25) 
intensities = s.get_lines_intensity()
intensities = [intens * mask for intens in intensities]
s.quantification_old(intensities=intensities)
