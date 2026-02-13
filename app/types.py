from pydantic import BaseModel, Field


class ReferenceRangeRule(BaseModel):
    """Rules to modulate reference ranges based on demographics.
    Uses simpleeval for conditions.
    """

    condition: str = Field(
        ..., description="Condition e.g. 'sex == female' or 'age > 18'"
    )
    min_normal: float | None = None
    max_normal: float | None = None
    priority: int = 0  # Higher priority rules applied last/overriding others


class BiomarkerEntry(BaseModel):
    """Core database entry for a known biomarker."""

    id: str = Field(
        ..., description="Canonical ID/Name of the biomarker (e.g. 'hemoglobin')"
    )
    aliases: list[str] = Field(
        default_factory=list, description="List of common names/aliases"
    )
    canonical_unit: str = Field(
        ..., description="The standard unit used for storage and comparison"
    )
    description: str | None = None

    # Validation ranges (generic, could be refined by age/sex later)
    min_normal: float | None = None
    max_normal: float | None = None

    # Mapping of other units to the canonical unit
    # Key is the unit name (e.g. 'mg/dL'), value is a formula string using 'x' (e.g. 'x / 38.67')
    conversions: dict[str, str] = Field(default_factory=dict)

    # Demographic-specific range rules
    reference_rules: list[ReferenceRangeRule] = Field(default_factory=list)

    # Metadata for where this entry came from (e.g. 'seed', 'research-agent')
    source: str = "seed"


class ExtractedBiomarker(BaseModel):
    """Raw extraction from the LLM/Vision model."""

    raw_name: str
    value: float
    unit: str
    flags: list[str] = Field(
        default_factory=list, description="Any H/L flags extracted"
    )
    confidence: float = Field(
        default=1.0, description="Confidence score of extraction if available"
    )


class AnalyzedBiomarker(BaseModel):
    """Final fully processed biomarker for the report."""

    biomarker_id: str
    display_name: str
    value: float
    unit: str
    status: str = "normal"  # normal, high, low, unknown
    notes: str | None = None
