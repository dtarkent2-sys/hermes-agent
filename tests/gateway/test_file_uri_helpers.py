"""Unit tests for file:// URI canonicalization helpers (gateway/platforms/base).

Producers (gateway/run.py, gateway/kanban_watchers.py, base.py's own queued
delivery) emit ``local_path_to_file_uri``; every consumer that decodes a
``file://`` image URL uses ``file_uri_to_local_path``. Together they fix the
Windows media-delivery bug class where the historical producer form
``f"file://{quote(p)}"`` percent-encoded drive colons and separators
(``file://C%3A%5CUsers%5C...``) and consumers hand-sliced ``uri[7:]``.
"""

from pathlib import Path

import pytest

from gateway.platforms.base import file_uri_to_local_path, local_path_to_file_uri


class TestLocalPathToFileUri:
    def test_posix_absolute(self):
        # On Windows a leading-/ path is drive-relative and resolves against
        # the current drive (Path semantics); on POSIX it is already absolute.
        uri = local_path_to_file_uri("/home/user/x/y.png")
        decoded = file_uri_to_local_path(uri)
        assert Path(decoded).is_absolute()

    def test_windows_absolute_canonical_form(self):
        assert local_path_to_file_uri("C:/Users/x/y.png") == "file:///C:/Users/x/y.png"

    def test_windows_backslash_input_same_uri(self):
        assert (
            local_path_to_file_uri("C:\\Users\\x\\y.png")
            == local_path_to_file_uri("C:/Users/x/y.png")
        )

    def test_spaces_percent_encoded(self):
        uri = local_path_to_file_uri("/home/my dir/x.png")
        assert "%20" in uri
        assert " " not in uri

    def test_relative_path_resolved_not_raised(self):
        uri = local_path_to_file_uri("relative/x.png")
        assert uri.startswith("file://")
        assert Path(file_uri_to_local_path(uri)).is_absolute()

    def test_round_trip_through_decoder(self, tmp_path):
        real = tmp_path / "img_0.png"
        real.write_bytes(b"x")
        assert Path(file_uri_to_local_path(local_path_to_file_uri(str(real)))) == real


class TestFileUriToLocalPath:
    def test_canonical_posix(self):
        assert file_uri_to_local_path("file:///home/user/x/y.png") == "/home/user/x/y.png"

    def test_canonical_windows_drive_strips_leading_slash(self):
        # The canonical as_uri() form puts the drive below a leading slash;
        # a bare slice would yield /C:/... which Windows resolves as \C:\...
        assert file_uri_to_local_path("file:///C:/Users/x/y.png") == "C:/Users/x/y.png"

    def test_legacy_quoted_windows_form(self):
        # Historical producer output: file://C%3A%5CUsers%5Cx%5Cy.png
        assert file_uri_to_local_path("file://C%3A%5CUsers%5Cx%5Cy.png") == "C:\\Users\\x\\y.png"

    def test_legacy_quoted_spaces(self):
        assert (
            file_uri_to_local_path("file:///home/my%20dir/x.png")
            == "/home/my dir/x.png"
        )

    def test_bare_unquoted_windows_path_all_backslashes_in_netloc(self):
        # f"file://{p}" on Windows: everything before the final separator is
        # the netloc; must decode to the full path, not drop the remainder.
        uri = "file://C:\\Users\\u\\AppData\\img.png"
        assert file_uri_to_local_path(uri) == "C:\\Users\\u\\AppData\\img.png"

    def test_bare_unquoted_windows_path_with_slash_tail(self):
        # f"file://{tmp_path}/missing.png": netloc ends at the first /, and
        # the tail keeps its original separator (legacy unquote behavior;
        # Windows Path accepts mixed separators).
        uri = "file://C:\\Users\\u\\tmpdir/missing_a.png"
        assert file_uri_to_local_path(uri) == "C:\\Users\\u\\tmpdir/missing_a.png"

    def test_bare_drive_slash_form(self):
        assert file_uri_to_local_path("file://C:/Users/x/y.png") == "C:/Users/x/y.png"

    def test_localhost_authority_treated_as_path(self):
        assert file_uri_to_local_path("file://localhost/home/x.png") == "/home/x.png"

    def test_unc_authority_preserved(self):
        assert (
            file_uri_to_local_path("file://server/share/x.png")
            == "//server/share/x.png"
        )

    def test_unc_canonical_form(self):
        # Path("//server/share/x.png").as_uri() == "file://server/share/x.png"
        assert (
            file_uri_to_local_path(local_path_to_file_uri("//server/share/x.png"))
            == "//server/share/x.png"
        )

    def test_producer_consumer_round_trip_all_forms(self, tmp_path):
        real = tmp_path / "sub dir" / "img.png"
        real.parent.mkdir(parents=True)
        real.write_bytes(b"x")
        expected = str(real)
        for uri in (
            local_path_to_file_uri(str(real)),           # canonical (new producer)
            "file://" + str(real).replace("\\", "/"),    # bare f"file://{p}"
        ):
            assert Path(file_uri_to_local_path(uri)) == real, uri
        # legacy quoted producer form decodes to the same path
        import urllib.parse

        legacy = "file://" + urllib.parse.quote(str(real))
        assert Path(file_uri_to_local_path(legacy)) == real
        assert expected  # silence unused-var linters

    def test_non_file_uri_passthrough_shape(self):
        # The function is only called for file:// URIs by consumers, but must
        # not raise on other shapes.
        assert file_uri_to_local_path("not-a-uri") == "not-a-uri"


class TestWindowsRealFiles:
    """End-to-end: decoded URI must point at a file that actually exists."""

    def test_canonical_uri_resolves_to_real_file(self, tmp_path):
        real = tmp_path / "media cache" / "photo.png"
        real.parent.mkdir(parents=True)
        real.write_bytes(b"png")
        uri = real.as_uri()
        decoded = Path(file_uri_to_local_path(uri))
        assert decoded == real
        assert decoded.is_file()

    def test_legacy_quoted_uri_resolves_to_real_file(self, tmp_path):
        import urllib.parse

        real = tmp_path / "media cache" / "photo.png"
        real.parent.mkdir(parents=True)
        real.write_bytes(b"png")
        uri = "file://" + urllib.parse.quote(str(real))
        decoded = Path(file_uri_to_local_path(uri))
        assert decoded == real
        assert decoded.is_file()

    def test_bare_unquoted_uri_resolves_to_real_file(self, tmp_path):
        real = tmp_path / "media cache" / "photo.png"
        real.parent.mkdir(parents=True)
        real.write_bytes(b"png")
        uri = f"file://{real}"
        decoded = Path(file_uri_to_local_path(uri))
        assert decoded == real
        assert decoded.is_file()


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("file:///a/b.png", "/a/b.png"),
        ("file:///C:/a/b.png", "C:/a/b.png"),
        ("file://C:/a/b.png", "C:/a/b.png"),
        ("file://C%3A%5Ca%5Cb.png", "C:\\a\\b.png"),
        ("file://localhost/a/b.png", "/a/b.png"),
        ("file://srv/share/a.png", "//srv/share/a.png"),
    ],
)
def test_decoder_table(uri, expected):
    assert file_uri_to_local_path(uri) == expected