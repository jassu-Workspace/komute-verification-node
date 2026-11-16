import base64
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.core.config import Settings


def _key(n: int = 32) -> str:
    return base64.b64encode(b"k" * n).decode()


def make(**kw) -> Settings:
    base = dict(
        api_key="a" * 32, storage_root=Path("/var/lib/komute"),
        storage_key_b64=_key(), storage_key_id="s1",
        manifest_key_b64=_key(), manifest_key_id="m1",
    )
    base.update(kw)
    return Settings(_env_file=None, **base)


def test_defaults_ok():
    s = make()
    assert s.env == "dev" and s.face_match_threshold == 0.35


def test_keys_must_decode_to_32_bytes():
    with pytest.raises(ValidationError):
        make(storage_key_b64=_key(16))
    with pytest.raises(ValidationError):
        make(manifest_key_b64="not base64!!")


def test_production_rejects_dev_skip_liveness():
    with pytest.raises(ValidationError):
        make(env="production", dev_skip_liveness=True, liveness_model=Path("m.onnx"))


def test_production_requires_liveness_model():
    with pytest.raises(ValidationError):
        make(env="production")
    assert make(env="production", liveness_model=Path("m.onnx")).env == "production"


@pytest.mark.parametrize("p", [
    r"C:\Users\x\OneDrive\Desktop\store", r"C:\Users\x\OneDrive - Corp\store", "/home/x/onedrive/s",
])
def test_storage_root_inside_onedrive_rejected(p):
    with pytest.raises(ValidationError):
        make(storage_root=Path(p))


def test_api_key_optional_in_dev_required_in_production():
    assert make(api_key=None).api_key is None
    assert make(api_key="").api_key is None
    with pytest.raises(ValidationError):
        make(env="production", liveness_model=Path("m.onnx"), api_key=None)


def test_production_rejects_short_api_key_dev_allows_it():
    with pytest.raises(ValidationError):
        make(env="production", liveness_model=Path("m.onnx"), api_key="short")
    assert make(api_key="short").env == "dev"
    assert make(env="production", liveness_model=Path("m.onnx"), api_key="k" * 24).env == "production"


def test_new_tunables_default_and_are_bounded():
    s = make()
    assert (s.name_match_threshold, s.min_age, s.plate_detect_score, s.barcode_budget_s) == (90.0, 18, 0.35, 10.0)
    for bad in ({"face_match_threshold": 0.01}, {"min_age": 5}, {"name_match_threshold": 10}, {"barcode_budget_s": 0}):
        with pytest.raises(ValidationError):
            make(**bad)


def test_values_load_from_dotenv_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text(
        f"KOMUTE_STORAGE_ROOT=/var/lib/komute\nKOMUTE_STORAGE_KEY_B64={_key()}\nKOMUTE_STORAGE_KEY_ID=s1\n"
        f"KOMUTE_MANIFEST_KEY_B64={_key()}\nKOMUTE_MANIFEST_KEY_ID=m1\nKOMUTE_FACE_MATCH_THRESHOLD=0.55\n"
    )
    assert Settings(_env_file=f).face_match_threshold == 0.55


def test_blank_liveness_model_line_means_none():
    assert make(liveness_model="").liveness_model is None
