from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """Base model for all COAST-Sim configuration classes.

    Enables ``validate_assignment`` so that field-level validators run on
    post-construction attribute assignment as well as at construction time.
    Forbids unrecognized fields so typo'd keys in mission YAML/JSON raise
    instead of being silently dropped.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
