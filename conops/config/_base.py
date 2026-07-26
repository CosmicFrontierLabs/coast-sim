from pydantic import BaseModel, ConfigDict, model_validator


class ConfigModel(BaseModel):
    """Base model for all COAST-Sim configuration classes.

    Enables ``validate_assignment`` so that field-level validators run on
    post-construction attribute assignment as well as at construction time.
    Forbids unrecognized fields so typo'd keys in mission YAML/JSON raise
    instead of being silently dropped.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_synthetic_null_name(cls, data: object) -> object:
        """Accept the empty ``name`` key emitted by the legacy YAML writer.

        Older annotated YAML output added ``name: null`` to every model in a
        list, including models without a ``name`` field. Keep strict handling
        for non-null values and every other unknown field.
        """
        if (
            isinstance(data, dict)
            and "name" not in cls.model_fields
            and "name" in data
            and data["name"] is None
        ):
            cleaned = dict(data)
            del cleaned["name"]
            return cleaned
        return data
