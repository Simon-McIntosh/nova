"""Manage IMAS Uniform Resource Identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.imas.dataentry import DataEntry

import numpy as np


@dataclass
class URI:
    """
    Manage IMAS Uniform Resource Identifiers.

    Follows the URI standard definition from RFC-3986 but is not fully compliant.
    The general URI structure is the following:

        - scheme:[//authority/]path[?query][#fragment].

    For sake of clarity and coherence, it was decided to define a single unified scheme
    for IMAS data resources (named imas) instead of defining different scheme for each
    backend. This implies that the backend needs to be specified in another manner.
    We opt for using the path part of the URI to specify the backend.

    As a result, the structure of the IMAS URI is the following, with elements between
    square brackets being optional:

    imas:[//host/]backend?query[#fragment]

    Each part of the URI are described in more details in the following subsections.

    Parameters
    ----------
    scheme: {'imas'}, optional
        An imas data entry is identified with 'imas'.

    host: str, optional
        Specify the address of the server on which the data is accessed.
        The structure of the host is '[user@]server[:port]', where:

    user is the username which will be recognized on the server to authenticate the
    submitter to this request. This information is optional, for instance for if the
    authentication is done by other means
    (e.g. using PKI certificates in the case of UDA) or if the data server does not
    require authentication;
    server is the address of the server (typically the fully qualified domain name or
                                         the IP of the server);
    port is optional and can be used to specify a port number onto which sending the
    requests to the server.
    When the data is stored locally the host (localhost) is omitted.

    Example: a host would typically be the address of a UDA server, with which the UDA
    backend of the Access-Layer will send requests for data over the netwrok. A URI
    would then look like: imas://uda.iter.org/uda?....

    Backend
    The backend is the name of the Access-Layer backend used to retrieve the stored
    data, this name is given in lower case and is mandatory. Current possibilities are:
        mdsplus, hdf5, ascii, memory and uda. Be aware that some backends may not be
        available in a given install of the Access-Layer.

    Query
    A query is mandatory. It starts with ? and is composed of a list of semi-colon ;
    (or ampersand &) separated pairs key=value. The following keys are standard and
    recognized by all backends:

    path: absolute path on the localhost where the data is stored;
    shot, run, user, database, version: allowed for compatibility purpose with legacy
    data-entry identifiers.
    Note: if legacy identifiers are provided, they are always transformed into a
    standard path before the query is being passed to the backend.

    Other keys may exist, be optional or mandatory for a given backend. Please refer
    to the latest documentation of the Access-Layer for more information on
    backend-specific keys.

    Fragment
    In order to identify a subset from a given data-entry, a fragment can be added
    to the URI. Such fragment, which starts with a hash #, is optional and
    allows to identify a specific IDS, or a part of an IDS.

    The structure of the fragment is #idsname[:occurrence][/idspath], where:

    idsname is the type name of the IDS, given in lower case, is mandatory in fragments
    and comes directly after the # delimiter;
    occurrence is the occurrence of the IDS
    (refer to the Access-Layer User Guide for more information), is optional and
    comes after a colon : delimiter that links the occurrence to the IDS specified
    before the delimiter;
    idspath is the path from IDS root to the IDS subset that needs to be identified,
    and is optional (in such case the fragment identifies the entire IDS structure).
    Refer to the IDS path syntax document for more information.
    """

    uri: str = ""

    def __call__(self, ids: DataEntry):
        """Return uri from data entry."""
        raise NotImplementedError

    def __getitem__(self, key: str):
        """Return item from query."""
        item = self.query[key]
        if key in ["pulse", "run"]:
            return int(item)
        return item

    @property
    def scheme(self) -> str:
        """Return uri scheme."""
        return self.uri.split(":", 2)[0]

    @property
    def authority(self) -> str:
        """Return uri authority."""
        if "//" in self.uri:
            return self.uri.split("//", 2)[1].split("/", 2)[0]
        return None

    @property
    def path(self):
        """Return uri path."""
        return self.uri.split("?")[0].split("#")[0].split("/")[-1].split(":")[-1]

    @cached_property
    def query(self) -> dict | None:
        """Return uri query."""
        if "?" in self.uri:
            return dict(
                zip(
                    *np.array(
                        [
                            pair.split("=")
                            for pair in self.uri.split("?")[-1].split("#")[0].split(";")
                            if len(pair) > 0
                        ]
                    ).T
                )
            )
        return None

    @cached_property
    def fragment(self) -> str | None:
        """Return uri fragment."""
        if "#" in self.uri:
            return self.uri.split("#")[1]
        return None

    @property
    def name(self) -> str | None:
        """Return ids name from fragment."""
        if self.fragment is None:
            return None
        return self.fragment.split(":")[0].split("/")[0]

    @property
    def occurrence(self) -> str | None:
        """Return ids occurence from uri fragment."""
        if self.fragment is None or ":" not in self.fragment:
            return None
        return self.fragment.split(":")[1].split("/")[0]

    @property
    def ids_path(self) -> str | None:
        """Return ids path from uri fragment."""
        if self.fragment is None or "/" not in self.fragment:
            return None
        return "/".join(self.fragment.split("/")[1:])
