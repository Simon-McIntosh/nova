import fabric
import paramiko
import pytest

jumpname = "sdcc-login.iter.org"
hostname = "sdcc-login04.iter.org"


def check_connect(hostname: str, jumpname: str) -> bool:
    """Return True if a ssh connection to hostname via jumphost is possible."""
    try:
        with fabric.Connection(hostname, gateway=fabric.Connection(jumpname)) as sdcc:
            assert (
                sdcc.run("hostname", hide=True, in_stream=False).stdout.strip()
                == hostname
            )
        return True
    except (
        AssertionError,
        paramiko.ssh_exception.ChannelException,
        paramiko.ssh_exception.socket.gaierror,
    ):
        return False


def mark_connect(hostname: str, jumpname: str) -> pytest.mark:
    """Return pytest mark for ssh connection to hostname via jumpname."""
    return pytest.mark.skipif(
        not check_connect(hostname, jumpname),
        reason=f"unable to esstablish ssh connection to {hostname} "
        f"via {jumpname} jumphost",
    )


ssh = mark_connect(hostname, jumpname)


@ssh
@pytest.mark.parametrize("hostname", ["sdcc-login03.iter.org", "sdcc-login04.iter.org"])
def test_check_connect(hostname):
    assert check_connect(hostname, "sdcc-login.iter.org")


if __name__ == "__main__":
    pytest.main([__file__])
