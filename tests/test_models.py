from urvc.models import BASE_MODEL, GIANT_MODEL, resolve_model


def test_model_presets_and_custom_model_passthrough():
    assert resolve_model("giant") == GIANT_MODEL
    assert resolve_model("default") == GIANT_MODEL
    assert resolve_model("base") == BASE_MODEL
    assert resolve_model("organization/custom-model") == "organization/custom-model"
