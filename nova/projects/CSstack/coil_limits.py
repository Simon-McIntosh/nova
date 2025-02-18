from nep.DINA.read_scenario import scenario_limits, scenario_data
from amigo.pyplot import plt

strID = "15MA DT-DINA"

plt.set_context("talk")

d2 = scenario_data()
folders = [f for f in d2.folders if strID in f][-12:]
folders = folders[:3] + folders[6:]
file = -1

limit = scenario_limits(folders[file], t="d3")
limit.load_data(folders[file])
limit.plot(multi=True, strID=strID)


ax = plt.subplots(len(limit.index), 1, sharex=True, sharey=True)[1]
for folder in folders:
    limit.load_data(folder)
    limit.plot(multi=False, strID=strID, ax=ax)
