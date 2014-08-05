"""
Visualisation

"""

pyplot.set_cmap('RdYlBu_r')

# 3D spectrum

s = database.spec3D('Ti_SEM')
im = database.image2D('Ti_SEM')

# elemental map and histogram
res = s.get_lines_intensity(plot_result=True)
utils.plot.plot_histograms(res)

# line scan
line_scan = s[::, 6.6]
res = line_scan.get_lines_intensity()
utils.plot.plot_spectra(res, legend='auto')

# Nav with SE image
dim = s.axes_manager.shape
im = im.rebin((dim[0], dim[1]))
s.plot_xray_lines('from_elements', navigator=im)

# 3D image

img = database.image3D()
s = database.result3D()
img2 = s.get_result('Ni', 'quant')

# Mayavi
fig, src, iso = img.plot_3D_iso_surface([0.2, 0.8])
fig, src2, iso2 = img2.plot_3D_iso_surface(0.2, figure=fig,
                                           outline=False)
iso2.contour.contours = [0.73, ]

# orthoview
image_eds.plot_orthoview_animated(img2)