
#font and size
font = {'family' : 'sans-serif',
    'sans-serif' : 'Tahoma',
    'size'   : 8}
rc('font', **font)
figsize(4,2.2) #4 == column
figsize(3.54,2.4) # exacttly 9 cm
figsize(5.5,2.4) # exacttly 14 cm
fig = figure()


#line style
colo = ['gray','black']
mark = ['o','+','x','^']
markeS= [2.5,4,4,3]
lst=['-','-','-','-']
lw=[1.5,1,0.6,0.6]
plot(xdat,ydat,
         color = colo[1],linewidth=0.6,marker= mark[i],
         ms=markeS[i],ls=lst[i])

fig = gcf()
ax = gca()

for li, lws in zip(ax.lines,lw):
    li.set_linewidth(lws)

#axes
xticks(frange(0,1001,200./n),frange(0,11,2./n))
ticklabel_format(style='sci', axis='y', scilimits=(0,0))
xlabel('')
ylabel('')

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
setp(ax.lines,linewidth=0.6)
setp(fig,facecolor=(1,1,1,1))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_position(matplotlib.transforms.Bbox(array([[0.125,0.13],[0.975,0.975]])))
#setp(fig,edgecolor=(0,0,0,0))
xlim(0,10)
ylim(0,220)
def price(x): return '$%1.2f'%x
ax.format_xdata = mdates.DateFormatter('%d-%m-%Y')
ax.format_ydata = price
fig

#2 y axes
ax2 = ax.twinx()
plot(xdat,ydat)
ax2.set_ylim(0,1)
ax2.set_ylabel('SE Contrast')
ax2.set_position(matplotlib.transforms.Bbox(array([[0.125,0.17],[0.9,0.975]])))


#subplot()
plt.subplots_adjust(bottom = 0.14)
for l, norm in enumerate([True,False]):
    for k, el in enumerate(['Ni','Co','all']):    
        subplot(range(231,237)[k+l*3])  

#inset
a = axes([.57, .62, .4, .33], axisbg='w')

plot(xdat,ydat)
ax = gca()
ylim(0,0.65e4)
xlim(3.2*scale_bins,15*scale_bins)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

#title('Noise Filtering with PCA')
#legend()
ax.legend_.set_frame_on(False)
ax.legend_.set_bbox_to_anchor((0.63, 0.55))

#annotate
wi=0.3
hw=3
fr=0.3
annotate('SE',xy=(0.45,1.025),xytext=(0.556, 1.07),
            arrowprops=dict(facecolor='black',width=wi, headwidth=hw,frac=fr))

#save
fig.savefig('FEMMS_.png',dpi=600)
savefig('images.svg', transparent=True)

getp(fig)
getp(ax)
