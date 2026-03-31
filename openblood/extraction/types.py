from __future__ import annotations

from pydantic import BaseModel, Field

from app.types import ExtractedBiomarker, ReportMetadata


class PageArtifact(BaseModel):
    page_num: int
    image_path: str | None = None
    text_layer: str | None = None


class EngineBiomarkerCandidate(BaseModel):
    raw_name: str
    value: float | str | bool
    unit: str
    raw_value_text: str | None = None
    specimen: str | None = None
    measurement_qualifier: str | None = None
    semantic_value: str | None = None
    is_computed_candidate: bool = False
    flags: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    page_num: int | None = None
    source_engine: str

    def to_extracted_biomarker(self) -> ExtractedBiomarker:
        return ExtractedBiomarker(
            raw_name=self.raw_name,
            value=self.value,
            unit=self.unit,
            raw_value_text=self.raw_value_text,
            specimen=self.specimen,
            measurement_qualifier=self.measurement_qualifier,
            semantic_value=self.semantic_value,
            is_computed_candidate=self.is_computed_candidate,
            flags=list(self.flags),
            confidence=self.confidence,
        )


class EngineExtractionResult(BaseModel):
    engine_id: str
    biomarkers: list[EngineBiomarkerCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: ReportMetadata = Field(default_factory=ReportMetadata)
    raw_payload: str = ""
    weight: float = 1.0


class ExtractionPipelineResult(BaseModel):
    biomarkers: list[ExtractedBiomarker] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: ReportMetadata = Field(default_factory=ReportMetadata)
    raw_payload: str = ""
    engine_results: list[EngineExtractionResult] = Field(default_factory=list)

    @classmethod
    def from_engine_result(
        cls,
        engine_result: EngineExtractionResult,
    ) -> ExtractionPipelineResult:
        return cls(
            biomarkers=[
                candidate.to_extracted_biomarker()
                for candidate in engine_result.biomarkers
            ],
            notes=list(engine_result.notes),
            metadata=engine_result.metadata.model_copy(deep=True),
            raw_payload=engine_result.raw_payload,
            engine_results=[engine_result],
        )
