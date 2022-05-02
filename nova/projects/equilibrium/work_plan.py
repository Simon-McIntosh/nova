"""Develop equilibrum reconstuction workplan."""
from dataclasses import dataclass, field
import operator

from datetime import datetime, timezone
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
             subtask_xticks=False):
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
            if isinstance(task, int):
                task = self.data.task[index].unique()[task]
            index &= self.data.task == task
            index &= self.data.subtask != task

        # phase index
        phase_index = (self.data.task == 'phase')
        phase_index &= (self.data.loc[phase_index, 'end'] >=
                        self.data.loc[index, 'start'].min())
        phase_index &= (self.data.loc[phase_index, 'start'] <=
                        self.data.loc[index, 'end'].max())
        index |= phase_index

        # select data
        data = self.data.loc[index, :].copy()
        labels = data[detail]

        # zero data offset
        zero_offset = data.start_offset[data.start.argmin()]
        data.loc[:, 'start_offset'] -= zero_offset
        data.loc[:, 'end_offset'] -= zero_offset

        data.loc[:, 'duration'] = [0 if label[:2] == 'ms' else duration
                                   for label, duration
                                   in zip(data.index, data.duration)]
        self.labels = labels
        width = 10 if task is None else 6
        ax = plt.subplots(1, 1, figsize=(width, len(labels.unique())/2.5))[1]
        ax.barh(labels, data.duration, left=data.start_offset,
                edgecolor='w', height=0.8)
        if milestones:
            milestone_data = data.loc[(data.duration == 0), :]
            if detail == 'task':
                ax.barh(milestone_data[detail], 0,
                        left=milestone_data.end_offset,
                        edgecolor='C3', facecolor='k',
                        linewidth=2, height=0.8)
            else:
                ax.plot(milestone_data.end_offset,
                        milestone_data[detail], 'C3d')

        plt.despine()

        for label in data.index[labels == 'phase']:
            ax.text(data.at[label, 'start_offset'] +
                    data.at[label, 'duration'] / 2, 0, label,
                    color='w', ha='center', va='center',
                    fontsize='x-small')

        for i, label in enumerate(labels.unique()[1:]):
            label_data = data.loc[(data[detail] == label)]
            end_offset = label_data.end_offset.max()
            duration = label_data.duration.sum()
            if duration > 0:
                ax.text(end_offset, i+1, f' {duration:1.0f}',
                        color='gray', ha='left', va='center',
                        fontsize='x-small')

        period = 24
        xticks = np.arange(0, data.end_offset.max()+1, period)
        xticks_labels = pandas.date_range(data.start.min(), end=data.end.max(),
                                          freq='M').strftime("%Y")
        xticks_labels = np.append(xticks_labels[::period], xticks_labels[-1])

        if len(xticks_labels) > len(xticks):
            xticks_labels = xticks_labels[:-1]
        if len(xticks) > len(xticks_labels):
            xticks = xticks[1:]

        xticks_minor = np.arange(0, data.end_offset.max(), int(period / 2))
        ax.set_xticks(xticks)
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(xticks_labels)
        yticks = ax.get_yticks()
        ax.set_yticks(yticks[1:])
        ax.invert_yaxis()

        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', which='both', length=0)

        if task is None:
            plt.title('overview')
        else:
            plt.title(task)
            if subtask_xticks:
                ax.set_xticks([])
                ax.tick_params(axis='x', which='both', length=0)
                ax.spines['bottom'].set_visible(False)

    def plot_subtasks(self):
        """Plot detailed subtask breakdown for all tasks."""
        for i in range(len(self.data.task.unique()))[1:]:
            self.plot(i)


if __name__ == '__main__':

    plan = WorkPlan()
    plan.plot()
    #plan.plot_subtasks()
