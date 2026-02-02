from app.core.config import settings

def test_model_path_exists():
    assert settings.MODEL_PATH is not None
    assert isinstance(settings.MODEL_PATH, str)
    