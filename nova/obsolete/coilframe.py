
from dataclasses import dataclass, field, InitVar
from typing import Union

from nova.frame.frame import Frame
from nova.frame.multipoint import MultiPoint


@dataclass
class CoilFrame:

    frame: Union[Frame, dict[str, Union[list, dict]]] = \
        field(default_factory=dict)
    metadata: InitVar[dict[dict[str, Union[list, dict]]]] = {}
    multipoint: MultiPoint = field(init=False)

    def __post_init__(self, metadata):
        if not isinstance(self.frame, Frame):
            self.frame = Frame(self.frame, metadata=metadata)
        else:
            self.frame.metadata = metadata
        self.multipoint = MultiPoint(self.frame)


if __name__ == '__main__':

    coilframe = CoilFrame({'x': 2, 'z': [5, 6, 7]},
                          metadata={'Required': ['x', 'z']})