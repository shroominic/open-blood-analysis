from typing import Literal

from pydantic import BaseModel, Field, model_validator


MeasurementQualifier = Literal[
    "exact",
    "below_limit",
    "above_limit",
    "below_detection",
    "approximate",
    "trace",
    "unknown",
]
BiomarkerKind = Literal["direct", "computed"]
InterpretationKind = Literal[
    "quantitative_range",
    "categorical_labels",
    "ordinal_labels",
    "computed_policy",
]


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


class LearnedContextAlias(BaseModel):
    raw_name: str
    raw_unit: str | None = None
    specimen: str | None = None
    representation: str | None = None
    confidence: float = 1.0
    source: str = "ai"


class LearnedValueAlias(BaseModel):
    raw_value: str
    semantic_value: str
    measurement_qualifier: MeasurementQualifier | None = None
    confidence: float = 1.0
    source: str = "ai"


class InterpretationPolicy(BaseModel):
    kind: InterpretationKind = "quantitative_range"
    label_map: dict[str, str] = Field(default_factory=dict)
    ordered_values: list[str] = Field(default_factory=list)


class ComputedDefinition(BaseModel):
    dependencies: list[str] = Field(default_factory=list)
    formula: str
    tolerance: float | None = None
    compute_when_missing: bool = False
    emit_when_reported: bool = True

    @model_validator(mode="after")
    def _validate_definition(self) -> "ComputedDefinition":
        if not self.formula.strip():
            raise ValueError("computed formula must not be empty")
        if not self.dependencies:
            raise ValueError("computed definition must declare dependencies")
        return self


class BiomarkerEntry(BaseModel):
    """Core database entry for a known biomarker."""

    id: str = Field(
        ..., description="Canonical ID/Name of the biomarker (e.g. 'hemoglobin')"
    )
    aliases: list[str] = Field(
        default_factory=list, description="List of common names/aliases"
    )
    kind: BiomarkerKind = "direct"
    analyte_family: str | None = None
    specimen: str | None = None
    representation: str | None = None
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
    interpretation: InterpretationPolicy | None = None
    computed_definition: ComputedDefinition | None = None
    learned_context_aliases: list[LearnedContextAlias] = Field(default_factory=list)
    learned_value_aliases: list[LearnedValueAlias] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_ranges(self) -> "BiomarkerEntry":
        if (
            self.min_normal is not None
            and self.max_normal is not None
            and self.min_normal > self.max_normal
        ):
            raise ValueError(
                f"min_normal ({self.min_normal}) must be <= max_normal ({self.max_normal})"
            )
        if (
            self.min_optimal is not None
            and self.max_optimal is not None
            and self.min_optimal > self.max_optimal
        ):
            raise ValueError(
                f"min_optimal ({self.min_optimal}) must be <= max_optimal ({self.max_optimal})"
            )
        if self.kind == "computed" and self.computed_definition is None:
            raise ValueError("computed biomarker entries require computed_definition")
        if self.representation is None:
            if self.kind == "computed":
                self.representation = "derived"
            elif self.value_type == "boolean":
                self.representation = "boolean"
            elif self.value_type == "enum":
                self.representation = "enum"
            else:
                self.representation = "quantitative"
        if self.interpretation is None:
            interpretation_kind: InterpretationKind = "quantitative_range"
            if self.kind == "computed":
                interpretation_kind = "computed_policy"
            elif self.value_type == "boolean":
                interpretation_kind = "categorical_labels"
            elif self.value_type == "enum":
                interpretation_kind = "ordinal_labels"
            self.interpretation = InterpretationPolicy(kind=interpretation_kind)
        return self


class ExtractedBiomarker(BaseModel):
    """Raw extraction from the LLM/Vision model."""

    raw_name: str
    value: float | str | bool
    unit: str
    raw_value_text: str | None = None
    specimen: str | None = None
    measurement_qualifier: MeasurementQualifier | None = None
    semantic_value: str | None = None
    is_computed_candidate: bool = False
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
    semantic_value: str | None = None
    measurement_qualifier: MeasurementQualifier | None = None
    provenance: Literal["observed", "computed", "verified_computed"] = "observed"
    derived_from: list[str] = Field(default_factory=list)
    reference_status: str | None = None
    optimal_status: str | None = None
    notes: str | None = None
    min_reference: float | None = None
    max_reference: float | None = None
    min_optimal: float | None = None
    max_optimal: float | None = None
    peak_value: float | None = None
