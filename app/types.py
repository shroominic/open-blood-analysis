from typing import Literal

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

    # Longevity-focused "optimal" range.
    # Leave null when unknown or effectively identical to the normal range.
    min_optimal: float | None = None
    max_optimal: float | None = None

    # Optional pinnacle healthy performance value.
    # Only set when a meaningful historical/elite healthy target exists.
    peak_value: float | None = None

    # Value semantics
    # - quantitative: numeric lab value
    # - boolean: binary qualitative value (e.g. Positive/Negative)
    # - enum: multi-class qualitative value (e.g. Negative/Trace/1+/2+/3+)
    value_type: Literal["quantitative", "boolean", "enum"] = "quantitative"
    enum_values: list[str] | None = None

    # Optional molecular weight used for mass<->molar concentration conversions.
    # Unit: g/mol.
    molar_mass_g_per_mol: float | None = None

    # Mapping of other units to the canonical unit
    # Key is the unit name (e.g. 'mg/dL'), value is a formula string using 'x' (e.g. 'x / 38.67')
    conversions: dict[str, str] = Field(default_factory=dict)

    # Demographic-specific range rules
    reference_rules: list[ReferenceRangeRule] = Field(default_factory=list)

    # Metadata for where this entry came from (e.g. 'seed', 'research-agent')
    source: str = "seed"

    # For qualitative biomarkers (e.g. "Negative", "Positive")
    # If set, any value in this list is considered Normal. Any value NOT in this list is High/Abnormal.
    normal_values: list[str] | None = None


class ExtractedBiomarker(BaseModel):
    """Raw extraction from the LLM/Vision model."""

    raw_name: str
    value: float | str | bool
    unit: str
    flags: list[str] = Field(
        default_factory=list, description="Any H/L flags extracted"
    )
    confidence: float = Field(
        default=1.0, description="Confidence score of extraction if available"
    )


class PatientInfo(BaseModel):
    age: int | None = None
    gender: str | None = None


class LabInfo(BaseModel):
    company_name: str | None = None
    location: str | None = None


class BloodCollectionInfo(BaseModel):
    date: str | None = None
    time: str | None = None
    datetime: str | None = None


class ReportMetadata(BaseModel):
    patient: PatientInfo = Field(default_factory=PatientInfo)
    lab: LabInfo = Field(default_factory=LabInfo)
    blood_collection: BloodCollectionInfo = Field(default_factory=BloodCollectionInfo)


class AnalyzedBiomarker(BaseModel):
    """Final fully processed biomarker for the report."""

    biomarker_id: str
    display_name: str
    value: float | str | bool
    unit: str
    status: str = "normal"  # e.g. low/high/optimal/normal_but_low_optimal
    reference_status: str | None = None
    optimal_status: str | None = None
    notes: str | None = None
    min_reference: float | None = None
    max_reference: float | None = None
    min_optimal: float | None = None
    max_optimal: float | None = None
    peak_value: float | None = None
