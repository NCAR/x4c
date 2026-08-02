'''Tests for the on-demand sample-case fetcher.

The tutorial notebooks run against a ~470 MB reduced CESM case. It is published as a
GitHub Release asset rather than committed, because the repository already carries a
4.7 GB `.git` and a Release asset does not weigh on `git clone`.

Everything here is offline: the download is monkeypatched, so no test needs the real
asset or network access.
'''
import hashlib
import os
import tarfile

import pytest

from x4c import utils


KEY = utils.DEFAULT_SAMPLE_CASE          # the short name callers pass, e.g. 'cesm1'
INFO = utils.SAMPLE_DATA[KEY]
CASE = INFO['case']                      # the case directory on disk
DATASET = INFO['dataset']                # its directory in the data repo and the cache
ARCHIVE = INFO['archive']


def _make_fake_archive(tmp_path, case_name=CASE, unsafe_member=None):
    """Build a tiny tar.gz shaped like the real sample archive."""
    tree = tmp_path / 'src' / case_name / 'atm' / 'proc' / 'tseries' / 'month_1'
    tree.mkdir(parents=True)
    (tree / 'placeholder.nc').write_bytes(b'not really netcdf')

    archive = tmp_path / ARCHIVE
    with tarfile.open(archive, 'w:gz') as tf:
        tf.add(tmp_path / 'src' / case_name, arcname=case_name)
        if unsafe_member is not None:
            info = tarfile.TarInfo(name=unsafe_member)
            info.size = 0
            tf.addfile(info)
    return archive


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at a scratch dir and clear the env overrides.

    Also drops the dataset's registered checksum: the archives built here are fakes and
    would never match the real asset. The tests that care about verification pass an
    explicit `sha256=`, or put one back into the registry.
    """
    monkeypatch.setenv('X4C_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.delenv('X4C_SAMPLE_DIR', raising=False)
    monkeypatch.delenv('X4C_SAMPLE_URL', raising=False)
    monkeypatch.setitem(utils.SAMPLE_DATA[KEY], 'sha256', None)
    return tmp_path


@pytest.fixture
def fake_download(monkeypatch, tmp_path):
    """Replace `utils.download` with a copy from a locally built archive."""
    archive = _make_fake_archive(tmp_path)
    calls = []

    def _dl(url, fname, **kws):
        calls.append(url)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(archive, 'rb') as src, open(fname, 'wb') as dst:
            dst.write(src.read())

    monkeypatch.setattr(utils, 'download', _dl)
    return calls, archive


# --------------------------------------------------------------------------- #
# resolution order
# --------------------------------------------------------------------------- #
def test_download_then_extract(isolated_cache, fake_download):
    calls, _ = fake_download
    path = utils.fetch_sample_data(verbose=False)

    assert os.path.isdir(path)
    assert os.path.basename(path) == CASE
    assert len(calls) == 1, 'should have downloaded exactly once'
    # the extracted tree is what matters; the tarball is not kept
    assert not os.path.exists(os.path.join(os.path.dirname(path), ARCHIVE))


def test_second_call_uses_the_cache(isolated_cache, fake_download):
    calls, _ = fake_download
    first = utils.fetch_sample_data(verbose=False)
    second = utils.fetch_sample_data(verbose=False)
    assert first == second
    assert len(calls) == 1, 'the cached copy should short-circuit the download'


def test_force_redownloads(isolated_cache, fake_download):
    calls, _ = fake_download
    utils.fetch_sample_data(verbose=False)
    utils.fetch_sample_data(verbose=False, force=True)
    assert len(calls) == 2


def test_sample_dir_env_takes_precedence(isolated_cache, monkeypatch, fake_download):
    """$X4C_SAMPLE_DIR should be used without any download."""
    calls, _ = fake_download
    local = isolated_cache / 'elsewhere' / CASE
    local.mkdir(parents=True)
    monkeypatch.setenv('X4C_SAMPLE_DIR', str(isolated_cache / 'elsewhere'))

    path = utils.fetch_sample_data(verbose=False)
    assert path == str(local)
    assert calls == [], 'should not download when a local copy is given'


def test_sample_dir_may_point_at_the_case_itself(isolated_cache, monkeypatch, fake_download):
    """Accept either the parent directory or the case directory."""
    local = isolated_cache / 'elsewhere' / CASE
    local.mkdir(parents=True)
    monkeypatch.setenv('X4C_SAMPLE_DIR', str(local))
    assert utils.fetch_sample_data(verbose=False) == str(local)


def test_bad_sample_dir_raises(isolated_cache, monkeypatch):
    monkeypatch.setenv('X4C_SAMPLE_DIR', str(isolated_cache / 'nope'))
    with pytest.raises(FileNotFoundError, match='X4C_SAMPLE_DIR'):
        utils.fetch_sample_data(verbose=False)


def test_url_env_override(isolated_cache, monkeypatch, fake_download):
    calls, _ = fake_download
    monkeypatch.setenv('X4C_SAMPLE_URL', 'https://example.invalid/custom.tar.gz')
    utils.fetch_sample_data(verbose=False)
    assert calls == ['https://example.invalid/custom.tar.gz']


# --------------------------------------------------------------------------- #
# integrity and safety
# --------------------------------------------------------------------------- #
def test_checksum_verified(isolated_cache, fake_download):
    _, archive = fake_download
    good = hashlib.sha256(open(archive, 'rb').read()).hexdigest()
    path = utils.fetch_sample_data(verbose=False, sha256=good)
    assert os.path.isdir(path)


def test_checksum_mismatch_raises_and_removes(isolated_cache, fake_download):
    """A corrupt download must not be left behind to poison the cache."""
    with pytest.raises(ValueError, match='Checksum mismatch'):
        utils.fetch_sample_data(verbose=False, sha256='0' * 64)
    cache = os.path.join(utils.cache_dir(), 'sample_data', DATASET)
    assert not os.path.exists(os.path.join(cache, ARCHIVE))


def test_registered_checksum_is_used_by_default(isolated_cache, monkeypatch, fake_download):
    """With no explicit sha256=, the registry's checksum must still be enforced."""
    monkeypatch.setitem(utils.SAMPLE_DATA[KEY], 'sha256', '0' * 64)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        utils.fetch_sample_data(verbose=False)


def test_unsafe_archive_paths_refused(isolated_cache, monkeypatch, tmp_path):
    """A tarball must not be able to write outside the extraction directory."""
    archive = _make_fake_archive(tmp_path / 'evil', unsafe_member='../escaped.txt')

    def _dl(url, fname, **kws):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(archive, 'rb') as s, open(fname, 'wb') as d:
            d.write(s.read())

    monkeypatch.setattr(utils, 'download', _dl)
    with pytest.raises(ValueError, match='unsafe path'):
        utils.fetch_sample_data(verbose=False)


def _serve_html(monkeypatch):
    """Stand in for GitHub answering a missing /raw/ path with 200 + an HTML page."""
    page = b'<!DOCTYPE html>\n<html lang="en">' + b' ' * 500

    def _dl(url, fname, **kws):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'wb') as f:
            f.write(page)

    monkeypatch.setattr(utils, 'download', _dl)


def test_html_page_is_not_cached_as_an_archive(isolated_cache, monkeypatch):
    """A 200 response that is not gzip must be rejected, not left in the cache."""
    _serve_html(monkeypatch)
    with pytest.raises(RuntimeError, match='did not return a gzip file'):
        utils.fetch_sample_data(verbose=False)
    cache = os.path.join(utils.cache_dir(), 'sample_data', DATASET)
    assert not os.path.exists(os.path.join(cache, ARCHIVE))


def test_html_page_is_not_cached_as_a_weight_file(isolated_cache, monkeypatch):
    """Same guard on the weight files: a wrong URL must not poison the cache."""
    _serve_html(monkeypatch)
    fname = 'map_not_a_real_grid_TO_1x1d_aave.nc.gz'
    with pytest.raises(RuntimeError, match='did not return a gzip file'):
        utils.fetch_wgt_file(fname, verbose=False)
    assert not os.path.exists(os.path.join(utils.cache_dir(), fname))


def test_archive_without_the_case_dir_raises(isolated_cache, monkeypatch, tmp_path):
    archive = _make_fake_archive(tmp_path / 'wrong', case_name='some_other_case')

    def _dl(url, fname, **kws):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(archive, 'rb') as s, open(fname, 'wb') as d:
            d.write(s.read())

    monkeypatch.setattr(utils, 'download', _dl)
    with pytest.raises(RuntimeError, match='did not contain'):
        utils.fetch_sample_data(verbose=False)


def test_http_error_message_is_actionable(isolated_cache, monkeypatch):
    """Until the Release asset exists, the error must say what to do instead."""
    import requests

    def _dl(url, fname, **kws):
        raise requests.HTTPError('404 Client Error: Not Found')

    monkeypatch.setattr(utils, 'download', _dl)
    with pytest.raises(RuntimeError) as ei:
        utils.fetch_sample_data(verbose=False)
    msg = str(ei.value)
    assert 'X4C_SAMPLE_DIR' in msg, 'should point at the local-copy escape hatch'
    assert '404' in msg


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def test_exported_at_top_level():
    import x4c
    assert x4c.fetch_sample_data is utils.fetch_sample_data


def test_cache_is_shared_with_the_weight_files(isolated_cache, fake_download):
    """Both fetchers live under the same cache root, so one env var relocates all."""
    root = utils.cache_dir()
    assert root == str(isolated_cache / 'cache')
    # the sample data goes in a per-dataset directory ...
    case_dir = utils.fetch_sample_data(verbose=False)
    assert case_dir == os.path.join(root, 'sample_data', DATASET, CASE)
    # ... the weight files directly in the root, both under the same relocatable root
    # (`fake_download` has replaced `utils.download`, so nothing is fetched for real)
    wgt = utils.fetch_wgt_file('map_not_a_real_grid_TO_1x1d_aave.nc.gz', verbose=False)
    assert wgt == os.path.join(root, 'map_not_a_real_grid_TO_1x1d_aave.nc.gz')


def test_registry_entries_are_well_formed():
    """Every dataset must carry the four fields the fetcher reads."""
    assert KEY in utils.SAMPLE_DATA
    for name, info in utils.SAMPLE_DATA.items():
        assert set(info) == {'dataset', 'case', 'tag', 'archive', 'sha256'}, name
        assert info['archive'].endswith('.tar.gz'), name
        assert info['sha256'] is None or len(info['sha256']) == 64, name


def test_url_is_built_from_the_registry():
    url = utils.sample_data_url(KEY)
    assert utils.DATA_REPO in url
    assert INFO['tag'] in url
    assert url.endswith(ARCHIVE)


def test_unknown_case_lists_the_available_ones():
    with pytest.raises(KeyError, match=KEY):
        utils.fetch_sample_data('cesm99', verbose=False)


def test_case_is_selectable_by_its_short_name(isolated_cache, fake_download):
    """The documented call is `fetch_sample_data(case='cesm1')`."""
    calls, _ = fake_download
    path = utils.fetch_sample_data(case=KEY, verbose=False)
    assert path.endswith(os.path.join(DATASET, CASE))
    assert len(calls) == 1


def test_sha256_helper(tmp_path):
    f = tmp_path / 'x.bin'
    f.write_bytes(b'hello world')
    assert utils._sha256(str(f)) == hashlib.sha256(b'hello world').hexdigest()
