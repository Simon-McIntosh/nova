
from matplotlib.lines import Line2D

from nova.electromagnetic.IO.read_scenario import scenario_data
from nova.utilities.pyplot import plt

d2 = scenario_data(read_txt=False)
scenarios = ['15MA DT-DINA2016-01_v1.1',
             '15MA DT-DINA2020-02', '15MA DT-DINA2020-04']

handels = []
for i, scenario in enumerate(scenarios):
    d2.load_file(scenario)
    d2.plot_current(color=f'C{i}')
    label = scenario.split('DINA')[-1]
    handels.append(Line2D([0], [0], color=f'C{i}', label=label))

plt.legend(handles=handels, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 1.14))
plt.xlabel('$t$ s')
plt.ylabel('$I$ kA')
plt.despine()
plt.xlim([0, 120])

plt.figure()
d2.plot_current()
plt.legend(loc='right', bbox_to_anchor=(1.3, 0.5))
plt.xlabel('$t$ s')
plt.ylabel('$I$ kA')
plt.despine()
plt.xlim([0, 120])