import os
from unittest.mock import patch


def test_config_loads_defaults():
    with patch.dict(os.environ, {"API_KEY": "test-key"}, clear=False):
        from importlib import reload
        import app.config as config_module
        reload(config_module)
        settings = config_module.Settings()
        assert settings.port == 8980
        assert settings.api_key == "test-key"
        assert settings.cpu_cores == 1


def test_config_loads_custom_values():
    env = {
        "PORT": "9000",
        "API_KEY": "my-secret",
        "CPU_CORES": "8",
    }
    with patch.dict(os.environ, env, clear=False):
        from importlib import reload
        import app.config as config_module
        reload(config_module)
        settings = config_module.Settings()
        assert settings.port == 9000
        assert settings.api_key == "my-secret"
        assert settings.cpu_cores == 8
