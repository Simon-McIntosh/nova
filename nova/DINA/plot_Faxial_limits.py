from nep.DINA.read_scenario import read_scenario

scn = read_scenario(read_txt=False)

ax = plt.subplots(sum(index), 1, sharex=True, sharey=True, figsize=(10, 6))[1]
#ax = [ax]
for i, folder in enumerate(folders['name'][index]):
    scn.load_file(folder=folder, read_txt=False)
    for j, gap in enumerate(scn.post['DINA']['Faxial'].columns[::-1]):
        ax[i].plot(scn.post['DINA']['t'], scn.post['DINA']['Faxial'][gap],
                   '-', color=f'C{j%10}', label=gap.replace('_', '-'))
    scn.get_max_value(scn.post['DINA']['t'],
                      scn.post['DINA']['Faxial'].loc[:, ::-1],
                      ax=ax[i], plot=True)
    '''
    ax[i].text(1.12, 0.95, folder, transform=ax[i].transAxes,
               ha='right', va='top', fontsize='x-small',
               bbox=dict(facecolor='w', ec='gray', lw=1,
                         boxstyle='round', pad=0.5))
    '''

    ax[i].set_ylim([0, 160])
    ax[i].set_ylabel('$F_{axial}^*$ MN')
ax[-1].set_xlabel('$t$ s')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
             ncol=4, fontsize='xx-small')
plt.despine()
plt.detick(ax)


