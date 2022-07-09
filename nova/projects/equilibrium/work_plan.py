"""Develop equilibrum reconstuction workplan."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from matplotlib.patches import Patch
import operator

from dateutil.relativedelta import relativedelta
import networkx
import numpy as np
import pandas

from nova.utilities.pyplot import plt


@dataclass
class WorkPlan:
    """
    Manage ITER equlibrium reconstruction workplan.

    All durations in months.
    """

    file: str = 'work_plan'
    data: pandas.DataFrame = field(default_factory=pandas.DataFrame)

    def __post_init__(self):
        """Load workplan."""
        self.read_excel()
        self.update_schedule()

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
        self.check_duration()

    def update_schedule(self):
        """Update time via directed graph topological sort."""
        self.data = self.update_time(self.data)
        self.data = self._sort()
        self.data = self._sort(reverse=True)
        self.data = self._sort()

        index = self.data.start.notna() & self.data.end.notna() & \
            self.data.duration.notna()
        start = self.data.start[index].min()
        self.data.loc[index, 'start_offset'] = \
            (self.data.start[index] - start).dt.days / (365.25 / 12)
        self.data.loc[index, 'end_offset'] = \
            self.data.loc[index, 'start_offset'] + self.data.duration[index]

    def _sort(self, reverse=False):
        """Order project phases."""
        data = self.data.copy()
        index = self.topological_sort(data)
        if reverse:
            index = index[::-1]
        for label in index:
            if data.loc[label, 'subtask'] == 'task':
                continue
            before = self.get_reference(
                data.at[label, 'before'], index, 'before')
            after = self.get_reference(
                data.at[label, 'after'], index, 'after')
            if not before and not after:
                continue
            if before:
                data.loc[label, 'end'] = data.at[before, 'start']
            elif after:
                data.loc[label, 'start'] = data.at[after, 'end']
            data.loc[label:, :] = self.update_time(data.loc[label:, :])
        return data

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
    def get_reference(reference: pandas.Series, index, side: str):
        """Return split reference list."""
        if isinstance(reference, str):
            labels = reference.split(',')
            if side == 'before':
                return next(label for label in index if label in labels)
            elif side == 'after':
                return next(label for label in index[::-1] if label in labels)
        return []

    def check_na(self):
        """Check labels for nans."""
        if (isnan := self.data.index.isna()).any():
            raise IndexError('nans present in index '
                             f'{self.data.loc[isnan, :]}')

    def check_unique(self):
        """Check that labels form a unique set."""
        unique_labels, counts = np.unique(self.data.index, return_counts=True)
        if (index := counts > 1).any():
            raise IndexError('check for duplicate labels '
                             f'{unique_labels[index]}')

    def check_duration(self):
        """Check for surficent information to define duration."""
        missing = self.data.duration.isna() & (self.data.subtask != 'task')
        if missing.any():
            raise IndexError(f'missing duration {self.data.duration[missing]}')

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
        labels = data.index
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
        if floating := [label for label, subtask, duration
                        in zip(labels, data.subtask, data.duration)
                        if label not in topo_index and subtask != 'task'
                        and duration != 0]:
            raise IndexError(f'floating labels {floating}')
        return topo_index

    def plot(self, task=None, milestones=True, n_ticks=12,
             subtask_xticks=False, axes=None, width=14,
             header_only=False):
        """Plot Gantt chart."""
        if milestones:
            index = self.data.duration.notna()
        else:
            index = (self.data.duration > 0)  # exclude milestones

        if task is None:
            detail = 'task'
            index &= self.data.task != 'phase'
        else:
            detail = 'subtask'
            match task:
                case str():
                    tasks = [task]
                case int():
                    tasks = [self.data.task[index].unique()[task]]
                case [*tasks]:
                    tasks = [self.data.task[index].unique()[task]
                             for task in tasks]
            task_index = np.zeros_like(index, dtype=bool)
            for task in tasks:
                task_index |= self.data.task == task
            index &= task_index
            index &= self.data.subtask != task

        # phase index
        phase_index = (self.data.task == 'phase')
        phase_index &= (self.data.loc[phase_index, 'end'] >=
                        self.data.loc[index, 'start'].min())
        phase_index &= (self.data.loc[phase_index, 'start'] <=
                        self.data.loc[index, 'end'].max())
        index |= phase_index
        header_index = ((self.data.subtask == 'phase') |
                        (self.data.subtask == 'assembly') |
                        (self.data.subtask == 'integrated commissioning'))
        if header_only:
            index &= header_index

        # select data
        data = self.data.loc[index, :].copy()
        data.loc[data.duration < 0, 'start_offset'] += \
            data.loc[data.duration < 0, 'duration']

        data.loc[:, 'duration'] = [0 if label[:2] == 'ms' else duration
                                   for label, duration
                                   in zip(data.index, data.duration)]
        data.loc[:, 'start_offset'] = [start_offset - 1e-1
                                       if label[:2] == 'ms' else start_offset
                                       for label, start_offset
                                       in zip(data.index, data.start_offset)]
        if task is not None:
            data = pandas.concat(
                [data.loc[header_index],
                 data.loc[~header_index].sort_values('start_offset')])

        labels = data[detail].copy()
        labels.loc[data.task == 'phase'] = 'phase'

        self.labels = labels
        if axes is None:
            axes = plt.subplots(1, 1,
                                figsize=(width, len(labels.unique())/4.5),
                                constrained_layout=~header_only)[1]

        data.loc[data.subtask == 'assembly', 'color'] = 'darkgray'
        data.loc[data.subtask == 'integrated commissioning', 'color'] = 'gray'
        data.loc[data.subtask == 'phase', 'color'] = 'black'
        data.loc[data.subtask == 'shutdown', 'color'] = 'darkgray'
        if task is None:
            data.loc[~header_index, 'color'] = 'C0'
        else:
            for i, task in enumerate(tasks):
                data.loc[data.task == task, 'color'] = f'C{i}'
        if task is not None and ((ntask := len(tasks)) > 1):
            axes.legend(handles=[Patch(facecolor=f'C{i}', label=task)
                                 for i, task in enumerate(tasks)],
                        loc='center', ncol=ntask,
                        bbox_to_anchor=(0.5, -0.08))

        axes.barh(labels, data.duration, left=data.start_offset,
                  color=data.color, edgecolor='w', height=0.8)
        if milestones:
            milestone_data = data.loc[(data.duration <= 0), :]
            if detail == 'task':
                axes.barh(milestone_data[detail], 0,
                          left=milestone_data.end_offset,
                          edgecolor='C3', facecolor='k',
                          linewidth=2, height=0.8)
            else:
                axes.plot(milestone_data.end_offset,
                          milestone_data[detail], 'd',
                          color='C3')
        plt.despine()

        for label in data.index[labels == 'phase']:
            axes.text(data.at[label, 'start_offset'] +
                      data.at[label, 'duration'] / 2, 0, label,
                      color='w', ha='center', va='center',
                      fontsize='xx-small')
        for i, label in enumerate(labels.unique()[1:]):
            label_data = data.loc[(data[detail] == label)]
            start_offset = label_data.start_offset.min()
            end_offset = label_data.end_offset.max()
            duration = label_data.duration.sum()
            if task is not None:
                axes.text(start_offset, i+1, f'{label}  ',
                          color='gray', ha='right', va='center',
                          fontsize='x-small')
            if duration > 0:
                axes.text(end_offset, i+1, f' {duration:1.0f}m',
                          color='gray', ha='left', va='center',
                          fontsize='x-small')
            else:
                date = self.data.start.min() + \
                    timedelta(end_offset * 365/12)
                axes.text(end_offset, i+1,
                          datetime.strftime(date, '  %m/%Y'),
                          color='gray', ha='left', va='center',
                          fontsize='x-small')

        self.set_xticks(data, axes)
        yticks = axes.get_yticks()
        axes.set_yticks(yticks[1:])
        axes.invert_yaxis()
        axes.spines['left'].set_visible(False)
        axes.tick_params(axis='y', which='both', length=0)
        if task is None:
            filename = 'overview'
        else:
            axes.set_yticks([])
            filename = '_'.join(tasks).replace(' ', '-')
        plt.savefig(f'{filename}.png')

    def set_xticks(self, data, axes, period=24):
        """Set axis xticks."""
        xticks = np.arange(data.start_offset.min(),
                           data.end_offset.max()+1, period)
        xticks_minor = np.arange(data.start_offset.min(),
                                 data.end_offset.max(), int(period / 2))
        xticks_range = pandas.date_range(
            data.start.min(), end=data.end.max() + pandas.offsets.MonthEnd(),
            freq='M')
        xticks_labels = xticks_range.strftime('%Y')[::period]

        if len(xticks_labels) > len(xticks):
            xticks_labels = xticks_labels[:-1]
        if len(xticks) > len(xticks_labels):
            xticks = xticks[1:]

        start_delta = \
            xticks_range[0] - datetime.strptime(xticks_labels[0], '%Y')
        start_offset = start_delta.days * 12 / 365 - 12
        xticks += start_offset
        xticks_minor += start_offset

        axes.set_xticks(xticks)
        axes.set_xticks(xticks_minor, minor=True)
        axes.set_xticklabels(xticks_labels)

    def plot_subtasks(self):
        """Plot detailed subtask breakdown for all tasks."""
        for i in range(len(self.data.task.unique()))[1:]:
            self.plot(i)

    @property
    def phase(self):
        """Return phase dataframe."""
        return self.data.loc[(self.data.task == 'phase')]

    def start_offset(self, data):
        """Return start offset."""
        return (data.start.min() - self.phase.start.min()).days * \
            12 / 365

    def resource(self):
        """Calculate and display overall ppy resource requirements."""
        data = self.data.loc[(self.data.task != 'phase') &
                             (self.data.duration > 0)]

        date_range = pandas.date_range(
            self.data.start.min(), self.data.end.max()
            + pandas.offsets.YearEnd(), freq='Y')
        composite = pandas.Series(0, index=date_range, dtype=int)
        for subtask in data.index:
            start = data.loc[subtask, 'start']
            end = data.loc[subtask, 'end']
            index = pandas.date_range(start, end, freq='M')
            series = pandas.Series(0, index=index, dtype=int).resample(
                'Y', convention='end').count()
            composite.loc[series.index] += series
        composite /= 12
        index = composite[composite != 0].index[-1]
        composite = composite.loc[:index]
        date_range = date_range[:composite.index.get_loc(index)+1]
        date_range -= date_range[0]
        date_range = date_range.days * 12 / 365

        axes = plt.subplots(2, 1, sharex=True, figsize=(14, 8),
                            gridspec_kw={'height_ratios': [1, 16],
                                         'hspace': 0})[1]
        self.plot(axes=axes[0], header_only=True)
        axes[1].bar(date_range, composite, width=10)
        for i in range(len(composite)):
            if composite[i] < 1:
                continue
            axes[1].text(date_range[i], composite[i],
                         f' {composite[i]:1.1f} ', rotation=90,
                         ha='center', va='top', color='w')
        self.set_xticks(data, axes[1])
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[0].spines['bottom'].set_visible(False)
        axes[1].set_ylabel('full-time equivalent')


if __name__ == '__main__':

    plan = WorkPlan()
    #plan.resource()
    plan.plot([1, 2, 3, 4])
    #plan.plot([7, 8, 9])
    #plan.plot_subtasks()
