import matplotlib.pyplot as plt
import numpy as np
from addtext import linelabel
import seaborn as sns
rc = {'figure.figsize':[3*3.14,3*3.14*7/12],'savefig.dpi':100,
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      "lines.linewidth": 5}
sns.set(context='poster',style='white',font='serif',palette='Set2',
        font_scale=1,rc=rc,)
color = sns.color_palette('Set2',5)
fs = 18

pfrac = 0.1

with plt.xkcd(scale=1):
    sns.set_context(rc={"lines.linewidth": 3})
    
    fig = plt.figure()
    text = linelabel(Ndiv=10,value='',loc='end')
    ax = fig.add_axes((0.1, 0.2, 0.6, 0.7))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xticks([])
    plt.yticks([])
    ax.set_ylim([0.6,1])

    R = np.linspace(1,1.5,100)
    pcore = (1+pfrac)/R
    prad = 0.76*np.ones(len(R))

    if pfrac == 0:
        plt.annotate('ne low for compatability',xy=(1.15,prad[0]*0.8),
                     arrowprops=dict(arrowstyle='->'), xytext=(1.02,prad[0]*0.95),
                     fontsize=fs)
        dlabel,clabel = '',''
    elif pfrac < 0:
        dlabel = ' moves inwards'
        clabel = ' {:+1.0f}%'.format(pfrac*100)
    else:
        dlabel = ' moves outwards'
        clabel = ' {:+1.0f}%'.format(pfrac*100)
    
 
    i = np.argmin(abs(pcore-prad))
    xd,yd = R[i],pcore[i]
    plt.annotate('detachment'+dlabel,xytext=(1.1,yd+0.2),xy=(xd,yd),
                 arrowprops=dict(arrowstyle='->'),
                 fontsize=fs)

    plt.plot(R,pcore,color=color[0])
    text.add('from core'+clabel,loc='end')

    
    plt.plot(R,prad,color=color[2])
    text.add('radiation',loc='end')
    if pfrac == 0: text.add('prad ~ ne ni Q(T)',loc='end')
    
    text.plot()
    plt.xlabel('Raduis')
    plt.ylabel('Power flux')
    

plt.savefig('../Figs/'+'xkcd_power_{:+1.0f}%'.format(pfrac*100))