import re

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

NO_DATA = "данных нет"


def numbers(text: str) -> set[float]:
    """All numbers in a text; '0,10' and '0.1' count as the same one."""
    return {float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", text)}


EMPTY = {"", "-", "—", "–", "n/a", "null", "none", "н/д", "не указано"}  

class ProductIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1)
    manufacturer: str | None = None
    specs: dict[str, str] = {}
    description: str | None = None

    @field_validator("specs", mode="before")
    @classmethod
    def clean_specs(cls, v):
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("specs must be an object")
        out = {}
        for k, x in v.items():
            key, val = str(k).strip(), ("" if x is None else str(x).strip())
            if key and val.lower() not in EMPTY:
                out[key] = val
        return out

    @property
    def source_text(self) -> str:
        return " ".join([self.name, self.manufacturer or "", self.description or "",
                         *self.specs.keys(), *self.specs.values()])


class LLMDraft(BaseModel):
    seo_title: str = Field(
        min_length=1, max_length=70,
        description="SEO title, max 70 characters: product name plus manufacturer if known.",
    )
    seo_description: str = Field(
        min_length=800, max_length=1200,
        description="SEO description, 800-1200 characters of flowing text, no markdown or lists. "
                    "If the input is sparse, reach the length with neutral wording about the device type, "
                    "without specific figures.",
    )
    search_queries: list[str] = Field(
        min_length=3, max_length=5,
        description="3-5 natural search queries the page should rank for.",
    )
    suitable_for: list[str] = Field(
        min_length=1, max_length=5,
        description='Who the device suits: up to 5 short items. Only what follows from the name or description; '
                    'if nothing can be inferred, a single item "данных нет".',
    )
    tasks: list[str] = Field(
        min_length=1, max_length=5,
        description='Tasks the device solves: up to 5 short items. Same rule as suitable_for.',
    )


class SeoResult(BaseModel):
    seo_title: str
    seo_description: str
    key_specs: dict[str, str]
    search_queries: list[str]
    commercial_block: dict[str, list[str]]

    @model_validator(mode="after")
    def numbers_come_from_input(self, info: ValidationInfo):
        allowed = (info.context or {}).get("allowed_numbers")
        if allowed is not None:
            invented = numbers(self.model_dump_json()) - allowed
            if invented:
                raise ValueError(f"numbers not in input: {sorted(invented)}")
        return self
