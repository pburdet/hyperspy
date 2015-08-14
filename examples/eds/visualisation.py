"""
Visualisation

"""

plt.rcParams['image.cmap'] = 'RdYlBu_r'
#plt.rcParams['image.cmap'] = 'gray'

#pyplot.set_cmap('RdYlBu_r')

# 3D spectrum

s = database.spec3D('Ti_SEM')
im = database.image2D('Ti_SEM')

# elemental map and histogram
res = s.get_lines_intensity(plot_result=True)
hs.plot.plot_histograms(res)

# line scan
line_scan = s[::, 6.6]
res = line_scan.get_lines_intensity()
hs.plot.plot_spectra(res, legend='auto')

# Nav with SE image
dim = s.axes_manager.shape
im = im.rebin((dim[0], dim[1]))
s.plot('from_elements', navigator=im)

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
img2.plot_orthoview()

#Modify param signal
img.plot(vmax = 0.45, vmin = 0.0)

#Modify param signal
img.plot(saturated_pixels=0.0)

#Draw an area with marker
splot =  database.image2D()
m = hs.plot.markers.rectangle(x1=10.,x2=20.,y1=10.,y2=15.,
                                 color='red')
splot.add_marker(m)
fig = plt.gcf()
fig

def fig_screenShot(f):
    data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(f.canvas.get_width_height()[::-1] + (3,))
    a = hs.signals.Spectrum(data)
    a.change_dtype('rgb8')
    return a
fig_screenShot(fig).plot()

#Change aspect ratio
im = database.result3D().metadata.Sample.quant[0].isig[:, 5.].as_image([0, 1])
im.plot()
im._plot.signal_plot.min_aspect = im.axes_manager.signal_axes[1].scale /\
	im.axes_manager.signal_axes[0].scale
im._plot.signal_plot.figure.clear()
im._plot.signal_plot.create_axis()
im._plot.signal_plot.plot()
