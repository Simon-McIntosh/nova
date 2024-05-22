"""Manage data pipeline."""

from dataclasses import dataclass, field


@dataclass
class PipeLine:
    """Pipeline datastructure for NakaData."""

    urls: list = field(default_factory=list)
    names: list = field(default_factory=list)
    count: int = 0

    def append(self, name, url):
        """Append name, url."""
        self.names.append(name)
        self.urls.append(url)
        self.count += 1

    def serve(self):
        """Return name, url tuple."""
        for name, url in zip(self.names, self.urls):
            yield name, url

    def flush(self):
        """Reset url pipeline."""
        self.__init__()
