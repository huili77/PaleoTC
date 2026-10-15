from pathlib import Path

import numpy as np
import pytest

from conftest import TRACKS
from paleotc.tracks import centers_by_time, read_tracks

REAL_TRACKS = Path(
    "/glade/campaign/cgd/ppc/brady/paleoweather/TCs/Eo3x/F3/tracks/TC_Eo3x.002_F3tracks_yrs31-60.txt"
)


def test_read_tracks_parses_points(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text(TRACKS)
    df = read_tracks(f)
    assert len(df) == 5
    assert df.track_id.tolist() == [0, 0, 1, 2, 3]
    assert df.iloc[0][["lon", "lat", "psl", "wind"]].tolist() == [0.0, 0.0, 9.9e4, 30.0]
    assert df.iloc[1][["year", "month", "day", "hour"]].tolist() == [31, 1, 1, 6]


def test_read_tracks_wraps_negative_longitude(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text(TRACKS)
    assert read_tracks(f).iloc[2].lon == pytest.approx(355.0)


def test_centers_by_time_groups_storms(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text(TRACKS)
    centers = centers_by_time(read_tracks(f), 31)
    assert set(centers) == {(31, 1, 1, 0), (31, 1, 1, 6), (31, 12, 31, 18)}
    np.testing.assert_allclose(
        sorted(map(tuple, centers[(31, 1, 1, 6)])), [(2.0, 0.5), (355.0, 15.0)]
    )
    assert centers_by_time(read_tracks(f), 32) == {}


@pytest.mark.glade
@pytest.mark.skipif(not REAL_TRACKS.exists(), reason="GLADE track file not available")
def test_real_track_file():
    df = read_tracks(REAL_TRACKS)
    assert df.track_id.nunique() == 2987
    assert set(df.hour.unique()) <= {0, 6, 12, 18}
    assert df.year.min() == 31 and df.year.max() == 61
