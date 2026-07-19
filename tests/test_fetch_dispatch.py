"""Dispatch must key on the parsed hostname, never a substring of the URL.

Regression cover for CodeQL ``py/incomplete-url-substring-sanitization`` (alerts
#1 and #2). The previous implementation tested ``"youtube.com" in url``, which
also matched attacker-controlled hosts and would have passed the URL to yt-dlp.

These tests never invoke the real fetchers — they monkeypatch them, so no network
access and no external CLIs are needed.
"""
import pytest

from honest_research import sources


@pytest.fixture(autouse=True)
def _stub_fetchers(monkeypatch):
    """Record which fetcher a URL dispatches to, without doing any work."""
    monkeypatch.setattr(sources, "fetch_youtube", lambda url: {"source": "youtube"})
    monkeypatch.setattr(sources, "fetch_tiktok", lambda url: {"source": "tiktok"})


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abc123",
    "https://youtube.com/watch?v=abc123",
    "http://m.youtube.com/watch?v=abc123",
    "https://youtu.be/abc123",
    "youtube.com/watch?v=abc123",          # scheme-less still resolves
    "HTTPS://WWW.YOUTUBE.COM/watch?v=A",   # case-insensitive
])
def test_genuine_youtube_hosts_dispatch(url):
    assert sources.fetch(url)["source"] == "youtube"


@pytest.mark.parametrize("url", [
    "https://www.tiktok.com/@user/video/123",
    "https://tiktok.com/@user/video/123",
    "tiktok.com/@user/video/123",
])
def test_genuine_tiktok_hosts_dispatch(url):
    assert sources.fetch(url)["source"] == "tiktok"


@pytest.mark.parametrize("url", [
    "https://youtube.com.evil.com/watch?v=a",   # suffix-confusion
    "https://evil.com/?next=youtube.com",       # substring in query
    "https://evil.com/youtube.com",             # substring in path
    "https://notyoutube.com/watch",             # no dot boundary
    "https://eviltiktok.com/@u/video/1",
    "https://evil.com/#tiktok.com",             # substring in fragment
    "https://evil.com/youtu.be/abc",
])
def test_spoofed_hosts_are_rejected(url):
    """Each of these passed the old substring check and reached a fetcher."""
    out = sources.fetch(url)
    assert out["source"] == "unknown"
    assert "unsupported url" in out["error"]


@pytest.mark.parametrize("url", ["", None, "   ", "not a url", "://broken"])
def test_degenerate_input_never_raises(url):
    assert sources.fetch(url)["source"] == "unknown"


def test_host_matcher_requires_a_dot_boundary():
    assert sources._host_matches("youtube.com", sources._YOUTUBE_HOSTS)
    assert sources._host_matches("www.youtube.com", sources._YOUTUBE_HOSTS)
    assert not sources._host_matches("notyoutube.com", sources._YOUTUBE_HOSTS)
    assert not sources._host_matches("youtube.com.evil.com", sources._YOUTUBE_HOSTS)
