from pathlib import Path
from PIL import Image
from soccer_gif_reid.prepare import scan


def test_full_release_reads_directory_id_for_query_and_named_reference(tmp_path):
    query = tmp_path / "train/player_0059/0052200111_00004845_5_g.jpg"
    reference = tmp_path / "gallery/player_0059/fred_vanvleet_raptors_65.jpg"
    for p in (query, reference):
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (30, 70)).save(p)
    assert scan(tmp_path / "train", "query", "train")[0]["pid"] == "59"
    assert scan(tmp_path / "gallery", "reference", "train")[0]["pid"] == "59"


def test_flat_release_identity_parsing_still_works(tmp_path):
    Image.new("RGB", (30, 70)).save(tmp_path / "0059_video_1.jpg")
    assert scan(tmp_path, "query", "train")[0]["pid"] == "59"
