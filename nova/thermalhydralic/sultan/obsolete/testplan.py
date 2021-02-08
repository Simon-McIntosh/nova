"""Manage Sultan testplan."""
from dataclasses import dataclass, field
from types import SimpleNamespace

from nova.thermalhydralic.sultan.campaign import Campaign


@dataclass
class Plan:
    """
    Load Sultan experiment test plan.

    Parameters
    ----------
    experiment : str
        Experiment label.

    """

    _experiment: str
    _index: int = 0
    test: Test = field(init=False)
    _campaign: Campaign = field(init=False, repr=False)
    reload: SimpleNamespace = field(init=False, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Init test campaign."""
        self.reload.__init__(experiment=True, index=True, data=True,
                             campaign=True)




if __name__ == '__main__':

    plan = Plan('CSJA_3', -1)
    print(plan.testindex)
    print(plan.testname)
    print(plan.plan['File'])
    print(plan.note)
    print(plan.mode)
