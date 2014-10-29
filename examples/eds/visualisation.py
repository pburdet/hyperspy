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

#Modify param signal
img.plot()
img._plot.signal_plot.auto_contrast = False
img._plot.signal_plot.vmax = 0.45
img._plot.signal_plot.vmin = 0.0
img._plot.signal_plot.update()

#Draw an area with marker
splot =  database.image2D()
pos = [10., 20., 10., 15.]
splot.plot()
m = utils.plot.marker()
m.type = 'line'
m.set_marker_properties(color='red')
for r in [0,1]:
    m.set_data(x1=pos[0],x2=pos[1],y1=pos[r+2],y2=pos[r+2])
    splot._plot.signal_plot.add_marker(m)
    m.plot()
    m.set_data(x1=pos[r],x2=pos[r],y1=pos[2],y2=pos[3])
    splot._plot.signal_plot.add_marker(m)
    m.plot()
fig = gcf()
fig
