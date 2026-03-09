from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """Base model for all COAST-Sim configuration classes.

    Enables ``validate_assignment`` so that field-level validators run on
    post-construction attribute assignment as well as at construction time.
    """

    model_config = ConfigDict(validate_assignment=True)
