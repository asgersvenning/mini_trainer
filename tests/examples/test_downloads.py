"""Failed downloads must not become reusable dataset archives."""

import io

import pytest

from examples import utils


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_download_failure_preserves_destination_and_allows_retry(tmp_path, monkeypatch, parallel, existing, error):
    destination = tmp_path / "images.tar.gz"
    if existing:
        destination.write_bytes(b"previous archive")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    payload = b"image bytes" * (1_000_000 if parallel else 3)
    failed = True

    class Response(io.BytesIO):
        def info(self):
            return {"Content-Length": str(len(payload)), "Accept-Ranges": "bytes" if parallel else "none"}

        def read(self, size=-1):
            if failed and self.tell():
                raise error("transfer interrupted")
            return super().read(size)

    def open_url(request):
        if request.get_method() == "HEAD":
            return Response()
        # A replacement must stay invisible until the entire transfer succeeds.
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
        return Response(payload)

    monkeypatch.setattr(utils.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    with pytest.raises(error, match="transfer interrupted"):
        utils.download_with_progress("https://example.invalid/images.tar.gz", destination)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    failed = False
    utils.download_with_progress("https://example.invalid/images.tar.gz", destination)
    assert list(tmp_path.iterdir()) == [destination]
    assert destination.read_bytes() == payload
