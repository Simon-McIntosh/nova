"""Develop equilibrum reconstuction workplan."""
from dataclasses import dataclass, field
import datetime
import operator
from typing import Union
from warnings import warn

from dateutil.relativedelta import relativedelta
import networkx
import numpy as np
import pandas
import plotly.express
import plotly.io


'''
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
'''

@dataclass
class Plotter:

    browser: bool = False

    def __post_init__(self):
        """Configure plot interface."""
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


@dataclass
class WorkPlan(Plotter):
    """Manage ITER equlibrium reconstruction workplan."""

    file: str = 'work_plan'
    data: pandas.DataFrame = field(default_factory=pandas.DataFrame)

    def __post_init__(self):
        """Load workplan."""
        self.read_excel()
        self.sort()

        #self.topological_sort(self.data)
        super().__post_init__()

    def __str__(self):
        """Return string representation of pandas dataframe."""
        return self.data.__str__()

    def read_excel(self):
        """Read plan data from excel."""
        self.data = pandas.read_excel(f'{self.file}.xlsx', index_col=[0, 1])
        self.data.drop(index='.', level=0, inplace=True)  # drop dot
        self.data.reset_index(inplace=True)
        self.data.set_index('label', inplace=True)
        self.check_na()
        self.check_unique()

    def sort(self, label=None):
        """Order project phases."""
        if label is not None:
            index = self.data['subtask'] == 'phase'
        else:
            index = self.data.index
        data = self.data.loc[index, :].copy()
        data.loc[data.index, :] = self.update_time(data)
        topo_index = self.topological_sort(data)

        for label in topo_index:
            if data.loc[label, 'subtask'] == 'task':
                continue
            before = self.get_reference_list(data.at[label, 'before'])
            after = self.get_reference_list(data.at[label, 'after'])
            reference = next(topo for topo in topo_index
                             if topo in before + after)
            if reference in before:
                data.loc[label, 'end'] = data.at[reference, 'start']
            elif reference in after:
                data.loc[label, 'start'] = data.at[reference, 'end']
            data.loc[label:, :] = self.update_time(data.loc[label:, :])

        print(data.start.to_dict())

    @staticmethod
    def update_time(data):
        """Update dataframe with endpoints calculated from duration."""
        data = data.copy()
        index = data.end.isna() & data.start.notna() & data.duration.notna()
        data.loc[index, 'end'] = WorkPlan.delta(data.loc[index, :], 'start')
        index = data.start.isna() & data.end.notna() & data.duration.notna()
        data.loc[index, 'start'] = WorkPlan.delta(data.loc[index, :], 'end')
        return data


    @staticmethod
    def get_reference_list(reference: pandas.Series):
        """Return split reference list."""
        if isinstance(reference, str):
            return reference.split(',')
        return []

    def check_na(self):
        """Check labels for nans."""
        if self.data.index.isna().any():
            raise IndexError('nans present in index '
                             f'{self.data.index.tolist()}')

    def check_unique(self):
        """Check that labels form a unique set."""
        unique_labels, counts = np.unique(self.data.index, return_counts=True)
        if (index := counts > 1).any():
            raise IndexError('check for duplicate labels '
                             f'{unique_labels[index]}')

    @staticmethod
    def delta(data: pandas.DataFrame, time: str) -> list:
        """Return list of datetimes."""
        if time == 'start':
            oper = operator.add
        elif time == 'end':
            oper = operator.sub
        else:
            raise NotImplementedError('time not in [start, end]')
        return [oper(time, relativedelta(months=month))
                for time, month in data.loc[:, [time, 'duration']].values]

    @staticmethod
    def check_reference(reference: pandas.Series):
        """Check referance tags are present in labels."""
        labels = reference.index.get_level_values(0)
        index = reference.notna()
        unique = np.unique([split for label in reference[index]
                            for split in label.split(',')])
        if notfound := [label for label in unique if label not in labels]:
            raise IndexError(f'references in [{reference.name}] not present '
                             f'in index {notfound}')

    @staticmethod
    def topological_sort(data: pandas.DataFrame):
        """Return topological sorted index calculated from dataframe."""
        labels = data.index.get_level_values(0)
        WorkPlan.check_reference(data['before'])
        WorkPlan.check_reference(data['after'])
        # create directed graph
        graph = networkx.DiGraph()
        index = data.after.notna()
        graph.add_edges_from(
            [(split_after, label)
             for label, after in zip(labels[index],
                                     data.loc[index, 'after'].values)
             for split_after in after.split(',')])
        index = data.before.notna()
        graph.add_edges_from(
            [(label, split_before)
             for label, before in zip(labels[index],
                                      data.loc[index, 'before'].values)
             for split_before in before.split(',')])
        topo_index = list(networkx.topological_sort(graph))
        if floating := [label for label in labels
                        if label not in topo_index]:
            raise IndexError(f'floating labels {floating}')
        return topo_index

    '''
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
    '''

    def plot_phase(self):
        """Plot project phases."""
        for phase in self.phase_duration:
            self.add_task(phase, phase)
        fig = plotly.express.timeline(self.data, x_start='start',
                                      x_end='end', y='task', color='phase')
        fig.update_yaxes(autorange="reversed")
        fig.show()


if __name__ == '__main__':

    plan = WorkPlan(browser=False)
    #plan.add_task('as-built data capture', 'PFP', duration=3*18+6)

    #print(plan.data)
    #plan.plot_phase()



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
