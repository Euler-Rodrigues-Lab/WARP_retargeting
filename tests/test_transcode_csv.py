"""Keep paper input geometry and timing unchanged across device migrations."""
import builtins
from pathlib import Path

import numpy as np


def test_public_csv_transcode_matches_frozen_fixture(tmp_path, monkeypatch):
    real_import = builtins.__import__
    def public_only(name, *args, **kwargs):
        if name.split('.')[0] in {'projects', 'geo_kin_ref', 'xr_robot_teleop_server', 'torch'}:
            raise AssertionError(f'Unexpected device dependency: {name}')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', public_only)
    from warp_retargeting.transcode_csv import transcode
    fixtures = Path(__file__).resolve().parents[1] / 'fixtures/seed'
    source = fixtures / 'washing_dishes_R_004__A299.csv.gz'
    output = transcode(source, tmp_path/'frames.npz')
    with np.load(fixtures/'washing_dishes_R_004__A299.frames.npz') as expected, np.load(output) as actual:
        assert set(actual.files) == set(expected.files)
        assert int(actual['meta/n']) == 420
        for key in expected.files:
            if key in {'meta/source', 'meta/notes'}:
                continue
            if expected[key].dtype.kind in 'fiu':
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12,
                                           err_msg=key)
            else:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
