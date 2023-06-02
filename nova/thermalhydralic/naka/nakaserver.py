"""Manage access to Naka server."""
from dataclasses import dataclass, field

import mechanize
from bs4 import BeautifulSoup
import http.cookiejar
import ssl


@dataclass
class NakaServer:
    """Manage access to Naka server."""

    year: int
    url: str = "https://scml-svr.naka.qst.go.jp"
    browser: mechanize.Browser = field(init=False, repr=False)

    def __post_init__(self):
        """Configure browser."""
        ssl._create_default_https_context = ssl._create_unverified_context
        self._configure_browser()

    def _configure_browser(self):
        self.browser = mechanize.Browser()
        self.browser.set_cookiejar(http.cookiejar.CookieJar())
        self.browser.set_handle_robots(False)

    def __enter__(self):
        """Login to Naka website and navigate to year."""
        self.browser.open(f"{self.url}/index-e.php")
        self.browser.select_form(nr=1)
        self.browser.form["userid"] = "FG"
        self.browser.form["password"] = "edmly70a"
        self.browser.submit()
        self.browser.open(f"{self.url}/data/{self.year}")
        return self

    def __exit__(self, type, value, traceback):
        """Close browser."""
        self.browser.close()

    @property
    def links(self):
        """Return server links to datafiles."""
        html = self.browser.response().read()
        soup = BeautifulSoup(html, "html.parser")
        return soup.find_all("a")

    @property
    def index(self):
        """Return packaged urls and filenames."""
        files = {}
        for link in self.links:
            href = link.get("href")
            if href[-4:] == ".pdf":
                run = href.split("/")[-1][:-4]
            elif href[-4:] == ".csv":
                run = "_".join(href.split("_")[1:3])
            else:
                continue
            if run not in files:
                files[run] = {"urls": [], "names": []}
            files[run]["urls"].append(href)
            files[run]["names"].append(href.split("/")[1])
        return files
