"""Read HLO source stacks from the serialized compiler metadata.

Field numbers follow OpenXLA's HloModuleProto and OpMetadata schemas. Unknown
fields are retained as opaque bytes; no numerical program is interpreted here.
"""

from collections import defaultdict
from functools import lru_cache

from google.protobuf.internal.decoder import _DecodeVarint


def fields(data):
    values = defaultdict(list)
    offset = 0
    while offset < len(data):
        tag, offset = _DecodeVarint(data, offset)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = _DecodeVarint(data, offset)
        elif wire == 2:
            size, offset = _DecodeVarint(data, offset)
            value, offset = data[offset : offset + size], offset + size
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value, offset = data[offset : offset + size], offset + size
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        values[number].append(value)
    return values


def first(data, number, default=0):
    return data.get(number, [default])[0]


def metadata(path):
    module = fields(path.read_bytes())
    index = fields(first(module, 17, b""))
    files = [""] + [value.decode() for value in index[1]]
    functions = [""] + [value.decode() for value in index[2]]
    locations = [{}] + [fields(value) for value in index[3]]
    frames = [{}] + [fields(value) for value in index[4]]

    @lru_cache(None)
    def stack(frame):
        if not frame:
            return []
        item = frames[frame]
        location = locations[first(item, 1)]
        parent = first(item, 2)
        assert parent < frame, (frame, parent)
        return stack(parent) + [
            {
                "file": files[first(location, 1)],
                "function": functions[first(location, 2)],
                "line": first(location, 3),
            }
        ]

    instructions = {}
    for raw in module[3]:
        computation = fields(raw)
        for encoded in computation[2]:
            instruction = fields(encoded)
            meta = fields(first(instruction, 7, b""))
            name = first(instruction, 1, b"").decode()
            instructions[name] = {
                "op_name": first(meta, 2, b"").decode(),
                "stack": stack(first(meta, 15)),
                "opcode": first(instruction, 2, b"").decode(),
                "computation": first(computation, 1, b"").decode(),
            }
    assert instructions and any(item["stack"] for item in instructions.values())
    return instructions
