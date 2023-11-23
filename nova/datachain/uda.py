"""Manage UDA data requests."""
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
import logging
import os
import re

from async_property import async_cached_property
import nest_asyncio

from nova.utilities.importmanager import mark_import

with mark_import("codac_uda") as mark_uda:
    from uda_client_reader.uda_client_reader_python import UdaClientReaderPython

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@dataclass
class Diagnostic:
    """Map diagnostic name to UDA variable."""

    name: str
    intergral: bool = True
    group: str = field(init=False)
    category: str = field(init=False)
    index: int = field(init=False)

    def __post_init__(self):
        """Translate diagnostic name to uda variable."""
        self.group, self.category, self.index = self.split(self.name)

    @staticmethod
    def split(name: str) -> tuple[str, str, int]:
        """Return uda variable name from given diagnostic name."""
        match re.split("[:.-]", name):  # IDS
            case "55", str(group), "00", str(category), str(index):
                return group, category, int(index)
            case "D1", "H1", str(group), str(category), str(index):
                return group[:-2], category, int(index)
            case _:
                raise NotImplementedError(
                    f"Mapping for diagnostic name {name} " "not implemented."
                )

    @cached_property
    def uda_name(self):
        """Return uda attribute name."""
        return f"D1-H1-{self.group}00:{self.category}-{self.index}"

    @cached_property
    def ids_name(self):
        """Return ids attribute name."""
        return f"55.{self.group}.00-{self.category}-{self.index}"

    @property
    def field_name(self):
        """Return uda field name."""
        if self.intergral:
            return "adcI"
        return "adcP"

    @cached_property
    def variable(self):
        """Return uda variable including field name."""
        return f"{self.uda_name}/{self.field_name}"


@dataclass
class UdaInfo:
    """Resolve UDA Uniform Data Identifier and manage server metadata."""

    client: UdaClientReaderPython | tuple[str, int] = ("10.153.0.254", 3090)
    machine: str = "ITER"
    pulse_root: str = "PCS"
    pulse_id: int | str = "*"

    def __post_init__(self):
        """Connect client."""
        if not isinstance(self.client, UdaClientReaderPython):
            with self._catch_uda_error():
                self.client = UdaClientReaderPython(*self.client)

    @contextmanager
    def _catch_uda_error(self):
        """Catch UDA error flags and raise."""
        yield
        if (error_code := self.client.getErrorCode()) < 0:
            raise ConnectionError(
                f"UDAClientReaderPyton returned negative error code {error_code}\n"
                f"{self.client.getErrorMsg()}"
            )

    @property
    def pulse(self) -> str:
        """Manage the uri path."""
        return os.path.join(self.pulse_root, str(self.pulse_id))

    @pulse.setter
    def pulse(self, pulse: str):
        match pulse.rsplit("/", 1):
            case str(pulse_root), str(pulse_id):
                self.pulse_root = pulse_root
                self.pulse_id = int(pulse_id)
            case _:
                raise ValueError(f"Unable to split pulse into root/id {pulse}")

    @property
    def _uri(self) -> str:
        """Return uri without checks."""
        return f"{self.machine}:{self.pulse}"

    @property
    def uri(self) -> str:
        """Manage the instance's uri composed from cached machine and path attrs."""
        if "*" in self.pulse or "*" in self.machine:
            self.update_uri()
        return self._uri

    @uri.setter
    def uri(self, uri: str):
        match uri.split(":", 1):
            case str(machine), str(pulse):
                self.machine = machine
                self.pulse = pulse
            case _:
                raise ValueError(f"Unable to split uri into scheme:pulse {uri}.")

    def update_uri(self):
        """Update URI resolving wildcards."""
        with self._catch_uda_error():
            self.uri = self.client.getLastPulse2(self._uri)

    async def aupdate_uri(self):
        """Return async resolved URI."""
        await asyncio.to_thread(self.update_uri)

    @cached_property
    def variables(self) -> tuple[str]:
        """Return synchronous variables list."""
        return self.get_variables("*")

    def get_variables(self, pattern: str) -> tuple[str]:
        """Return synchronous variable list matching given pattern."""
        with self._catch_uda_error():
            return self.client.getVariableList(pattern)

    @async_cached_property
    async def avariables(self):
        """Return full variable list."""
        return await self.aget_variables("*")

    async def aget_variables(self, pattern: str):
        """Return awaitable variable list matching given pattern."""
        return await asyncio.to_thread(self.get_variables, pattern)

    async def async_variables(self):
        """Return async query generator."""
        for variable in self.variables[:3]:
            yield variable

    @cached_property
    def pulse_info(self):
        """Return pulse info."""
        with self._catch_uda_error():
            return self.client.getPulseInfo2(self.uri)

    @cached_property
    def timestamp_from(self):
        """Return pulse start timestamp."""
        return self.pulse_info.timeFrom

    @cached_property
    def timestamp_to(self):
        """Return pulse end timestamp."""
        return self.pulse_info.timeTo

    @async_cached_property
    async def apulse_info(self):
        """Perform asynchronous pulse info update."""
        pulse_info = await asyncio.to_thread(self.client.getPulseInfo2, self.uri)
        self.timestamp_from = pulse_info.timeFrom
        self.timestamp_to = pulse_info.timeTo
        return pulse_info


@dataclass
class UdaSample:
    """Manage sample parameters."""

    time: float = 0
    duration: float = 1
    sample_number: int = 1
    sample_type: str = "last"

    @property
    def relative_start_time(self):
        """Return relative start time."""
        return self.time

    @property
    def relative_end_time(self):
        """Return relative end time."""
        return self.time + self.duration

    @cached_property
    def sample(self) -> str:
        """Retun cached UDA sample parameters."""
        return (
            f"startTime={self.relative_start_time}S,"
            f"endTime={self.relative_end_time}S,"
            f"decSamples={self.sample_number},"
            f"decType={self.sample_type}"
        )


@dataclass
class UdaQuery(UdaInfo, UdaSample):
    """Construct UDA sample queries."""

    pattern: str = "*"

    def __post_init__(self):
        """Load variable generator."""
        super().__post_init__()
        self.generator = (variable for variable in self.variables[:10])

    def __call__(self, variable):
        """Return UDA query."""
        return f"variable={variable},pulse={self.pulse},{self.sample}"

    async def async_generator(self):
        """Return async variable generator."""
        for variable in self.generator:
            yield variable


@dataclass
class UdaBase:
    """UDA client base class."""

    host: str = "10.153.0.254"
    port: int = 3090
    client: UdaClientReaderPython = field(init=False, repr=False)
    handle: int = field(init=False, default=-1)

    async def __aenter__(self):
        """Connect to uda server."""
        self.client = await asyncio.to_thread(
            UdaClientReaderPython, self.host, self.port
        )
        await asyncio.to_thread(self.client.resetAll)
        logging.info("Connected to uda client.")
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        """Clear client and close connection."""
        if self.handle >= 0:
            await asyncio.to_thread(self.client.releaseData, self.handle)
        logging.info("Released data from uda client.")


@dataclass
class UdaClient(UdaBase):
    """Manage connection to uda server."""

    query: str = field(default="")

    async def __aenter__(self):
        """Connect to uda server and fetch data."""
        await super().__aenter__()
        logging.info(f"Processing query {self.query}")
        self.handle = await self.fetch_data(self.query)
        return self

    async def fetch_data(self, query: str):
        """Return UDA handle for variable request via clent's fetchData method."""
        handle = await asyncio.to_thread(self.client.fetchData, query)
        if handle >= 0:
            return handle
        error_message = await asyncio.to_thread(self.client.getErrorMsg)
        raise ConnectionError(
            "Unable to connect to UDA server.\n"
            f"{error_message}\n"
            f"host: {self.host}, port: {self.port}."
        )


async def publish(query: UdaQuery, queue: asyncio.Queue):
    """Publish items to queue."""
    async for variable in query.async_generator():
        await queue.put(query(variable))
        logging.info(f"Published variable {variable}")
        # await asyncio.sleep(0)


async def request(queue: asyncio.Queue):
    """Request items from queue."""
    while True:
        query = await queue.get()
        logging.info(f"Requested variable {query}")
        async with UdaClient(query=query) as uda:
            print(uda.handle)
        # await asyncio.sleep(0)
        queue.task_done()

    # async with UdaBase() as uda:
    #    logging.info("UDA client is live")
    #    print("*", uda.client.getLastPulse2("ITER:PCS/*"))
    #    print(uda.client.getLastPulse2("ITER:PCS/*"))


async def main():
    """Process diagnostic signals."""
    queue = asyncio.Queue()

    query = UdaQuery(pulse_id=62, duration=1.5)

    producers = [asyncio.create_task(publish(query, queue)) for _ in range(10)]
    consumers = [asyncio.create_task(request(queue)) for _ in range(1000)]

    await asyncio.gather(*producers)
    await queue.join()
    for consumer in consumers:
        consumer.cancel()


if __name__ == "__main__":
    import time

    start_time = time.perf_counter()
    # asyncio.run(main())
    print(f"run time {time.perf_counter() - start_time:1.3f}s")
