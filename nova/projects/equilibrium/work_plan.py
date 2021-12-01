"""Develop equilibrum reconstuction workplan."""
from dataclasses import dataclass, field
import datetime
from typing import Union

from dateutil.relativedelta import relativedelta as delta
import numpy as np
import pandas
import plotly.express
import plotly.io


@dataclass
class Phase:
    """Build phased approach timeline."""

    first_plasma: Union[str, datetime.date] = '2025-12-01'
    phase_duration: dict[str, int] = field(
        default_factory=lambda: dict(PFP=3*12, FP=18, PFPO1=18, PFPO2=21,
                                     FPO1=16+8, FPO2=16+8, FPO3=16+8))

    def __post_init__(self):
        """Build phased approach timeline."""
        if not isinstance(self.first_plasma, datetime.date):
            self.first_plasma = datetime.date.fromisoformat(self.first_plasma)
        duration = np.cumsum(list(self.phase_duration.values()))
        self.phase_end = {phase: self.first_plasma -
                          delta(months=duration[1]) +
                          delta(months=duration[i])
                          for i, phase in enumerate(self.phase_duration)}
        self.phase_start = {phase: self.phase_end[phase] -
                            delta(months=abs(self.phase_duration[phase]))
                            for phase in self.phase_duration}


@dataclass
class WorkPlan(Phase):
    """Manage ITER equlibrium reconstruction workplan."""

    browser: bool = False
    data: pandas.DataFrame = field(default_factory=pandas.DataFrame)

    def __post_init__(self):
        """Init plotly renderer."""
        super().__post_init__()
        self.set_browser(self.browser)

    @property
    def renderer(self):
        """Return plotly renderer string."""
        if self.browser:
            return 'browser'
        return 'svg'

    def set_browser(self, browser: bool):
        """Set plotly renderer."""
        self.browser = browser
        plotly.io.renderers.default = self.renderer

    def relative_date(self, months: float) -> datetime.date:
        """Return date relative to first plasma."""
        return self.first_plasma + delta(months=months)

    def add_task(self, task: str, phase: str, duration=None, offset=0):
        """Insert task to data."""
        start = self.phase_start[phase] + delta(months=offset)
        if duration is not None:
            end = start + delta(months=duration)
        else:
            end = self.phase_end[phase]
        self.data = self.data.append(
            dict(task=task, start=start, end=end, phase=phase),
            ignore_index=True)

    def plot_phase(self):
        """Plot project phases."""
        for phase in self.phase_duration:
            self.add_task(phase, phase)
        fig = plotly.express.timeline(self.data, x_start='start',
                                      x_end='end', y='task', color='phase')
        fig.update_yaxes(autorange="reversed")
        fig.show()


if __name__ == '__main__':

    #plan = WorkPlan(browser=False)
    #plan.add_task('as-built data capture', 'PFP', duration=3*18+6)

    #print(plan.data)
    #plan.plot_phase()

    plan = pandas.read_excel('work_plan.xlsx', index_col=[0, 1])

    print(plan.index)
    print(plan.columns)
    print(plan)

'''

start = datetime.date(2022, 1, 1)

end = start + delta(months=3)

print(end)


start_date = dict(assembly=datetime.date('2022-01-01',
             first_plasma='2022-')

gantt = pandas.DataFrame([
    dict(Task="Job A", start='0', end='18', Resource="Alex"),
    dict(Task="Job B", start='0', end='18', Resource="Alex"),
    dict(Task="Job C",  start='0', end='18', Resource="Max")
])

fig = plotly.express.timeline(gantt, x_start='start',
                              x_end='end', y="Resource", color="Resource")
fig.show()
'''
