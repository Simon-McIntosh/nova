from itertools import cycle

# from elliptic import grid
# import cross_coil as cc
import seaborn as sns

rc = {
    "figure.figsize": [10, 10 * 12 / 16],
    "savefig.dpi": 100,
    "savefig.jpeg_quality": 100,
    "savefig.pad_inches": 0.1,
    "lines.linewidth": 2,
}
sns.set(
    context="talk",
    style="white",
    font="sans-serif",
    palette="Set2",
    font_scale=7 / 8,
    rc=rc,
)
color = sns.color_palette("Set2")
Color = cycle(color)

with open("../../eqdsk/ITER/ITER_15MA.eqdat", "r") as f:
    line = "start"
    for i in range(50):
        line = f.readline()
        print(line[:-1])
        print(len(line))
    # print(f.readline())
    # print(f.readline())

"""
setup = Setup('tmp', eqdir='../../eqdsk/VDE_UP_slow_maxV-force/')
sf = SF(setup.filename)


sf.contour(boundary=False)
#pl.plot(sf.Mpoint[0], sf.Mpoint[1], 'o')
pl.plot(sf.Xpoint_array[0, 1], sf.Xpoint_array[1, 1], 'o')
pl.plot(sf.Xpoint_array[0, 0], sf.Xpoint_array[1, 0], 'o')
# eq = EQ(sf,sigma=0,limit=[5.5,12,-8,5],n=5e4)  # resample
# eq.plotj()

pl.plot(sf.eqdsk['rbdry'], sf.eqdsk['zbdry'])
pl.plot(sf.eqdsk['xlim'], sf.eqdsk['ylim'])

pl.plot(np.sqrt(8.8**2+12.6**2),-3.67, 'o')
"""
