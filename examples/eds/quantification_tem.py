
###Cliff-Lorimer

#simple method

s = database.spec3D('TEM')
s.set_elements(["Ni", "Cr",'Al'])
s.set_lines(["Ni_Ka", "Cr_Ka", "Al_Ka"])
kfactors = [s.metadata.Sample.kfactors[2],
       s.metadata.Sample.kfactors[6]]
intensities = s.get_two_windows_intensities(
     bck_position=[[1.2,3.0],[5.0,5.7],[5.0,9.5]])
res = s.quant_cliff_lorimer_simple(intensities,kfactors)
utils.plot.plot_signals(res)

#Simulate two elements standard
s.set_microscope_parameters(live_time=30)
s.simulate_two_elements_standard(nTraj=100)
s.get_kfactors_from_standard()
s.quant_cliff_lorimer()

#kfactors from first principles
s.get_kfactors_from_first_principles()
s.quant_cliff_lorimer()


