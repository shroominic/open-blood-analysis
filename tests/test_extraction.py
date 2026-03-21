import asyncio
import json

from app.config import Config, ExtractionEngineSpec
from app.extraction.engines.gemini_vision import GeminiVisionEngine
from app.extraction.orchestrator import extract_report
from app.extraction.types import EngineExtractionResult, PageArtifact
from app.types import ExtractedBiomarker, ReportMetadata


def test_gemini_vision_engine_wraps_llm_extract_biomarkers(monkeypatch):
    async def fake_extract_biomarkers(
        image_paths: list[str],
        config: Config,
        *,
        model: str | None = None,
        client=None,
    ):
        assert image_paths == ["/tmp/page_1.jpg", "/tmp/page_2.jpg"]
        assert isinstance(config, Config)
        assert model == "gemini-test"
        return (
            [
                ExtractedBiomarker(
                    raw_name="Glucose",
                    value=95.0,
                    unit="mg/dL",
                    flags=[],
                    confidence=0.8,
                )
            ],
            ["fasting sample"],
            ReportMetadata(),
            '{"data":[{"raw_name":"Glucose","value":95,"unit":"mg/dL","flags":[]}]}',
        )

    monkeypatch.setattr(
        "app.extraction.engines.gemini_vision.llm.extract_biomarkers",
        fake_extract_biomarkers,
    )

    result = asyncio.run(
        GeminiVisionEngine(model="gemini-test").extract(
            page_artifacts=[
                PageArtifact(page_num=1, image_path="/tmp/page_1.jpg"),
                PageArtifact(page_num=2, image_path="/tmp/page_2.jpg"),
            ],
            config=Config(),
        )
    )

    assert result.engine_id == "gemini_vision"
    assert result.notes == ["fasting sample"]
    assert result.raw_payload
    assert len(result.biomarkers) == 1
    assert result.biomarkers[0].raw_name == "Glucose"
    assert result.biomarkers[0].source_engine == "gemini_vision"
    assert result.biomarkers[0].confidence == 0.8


def test_extract_report_uses_supplied_engine_and_preserves_output():
    class StubEngine:
        engine_id = "stub"
        execution_mode = "document"

        def __init__(self) -> None:
            self.page_artifacts: list[PageArtifact] = []

        async def extract(
            self,
            *,
            page_artifacts: list[PageArtifact],
            config: Config,
        ) -> EngineExtractionResult:
            self.page_artifacts = page_artifacts
            return EngineExtractionResult(
                engine_id=self.engine_id,
                biomarkers=[],
                notes=["ok"],
                metadata=ReportMetadata(),
                raw_payload="{}",
            )

    engine = StubEngine()

    result = asyncio.run(
        extract_report(
            image_paths=["/tmp/a.jpg", "/tmp/b.jpg"],
            config=Config(),
            engines=[engine],
        )
    )

    assert [artifact.page_num for artifact in engine.page_artifacts] == [1, 2]
    assert [artifact.image_path for artifact in engine.page_artifacts] == [
        "/tmp/a.jpg",
        "/tmp/b.jpg",
    ]
    assert result.notes == ["ok"]
    raw_payload = json.loads(result.raw_payload)
    assert raw_payload["data"] == []
    assert raw_payload["engines"]["stub"]["notes"] == ["ok"]
    assert len(result.engine_results) == 1
    assert result.engine_results[0].engine_id == "stub"


def test_extract_report_fuses_multiple_engines():
    class BetterLabelsEngine:
        engine_id = "primary"
        execution_mode = "document"

        async def extract(
            self,
            *,
            page_artifacts: list[PageArtifact],
            config: Config,
        ) -> EngineExtractionResult:
            return EngineExtractionResult(
                engine_id=self.engine_id,
                weight=2.0,
                biomarkers=[
                    {
                        "raw_name": "Lead in Blood",
                        "value": 5.2,
                        "unit": "ug/L",
                        "source_engine": self.engine_id,
                    }
                ],
                notes=["from primary"],
                raw_payload='{"data":[{"raw_name":"Lead in Blood"}]}',
            )

    class NoisyEngine:
        engine_id = "secondary"
        execution_mode = "document"

        async def extract(
            self,
            *,
            page_artifacts: list[PageArtifact],
            config: Config,
        ) -> EngineExtractionResult:
            return EngineExtractionResult(
                engine_id=self.engine_id,
                weight=0.5,
                biomarkers=[
                    {
                        "raw_name": "RL 1PIC",
                        "value": 5.2,
                        "unit": "ug/L",
                        "source_engine": self.engine_id,
                    }
                ],
                notes=["from secondary"],
                raw_payload='{"data":[{"raw_name":"RL 1PIC"}]}',
            )

    result = asyncio.run(
        extract_report(
            image_paths=["/tmp/a.jpg"],
            config=Config(extraction_fusion_mode="union"),
            engines=[BetterLabelsEngine(), NoisyEngine()],
        )
    )

    assert [biomarker.raw_name for biomarker in result.biomarkers] == ["Lead in Blood"]
    assert result.notes == ["from primary", "from secondary"]
    raw_payload = json.loads(result.raw_payload)
    assert set(raw_payload["engines"]) == {"primary", "secondary"}


def test_extract_report_merges_page_mode_engine_results():
    class PageEngine:
        engine_id = "page_stub"
        execution_mode = "page"

        def __init__(self) -> None:
            self.calls: list[list[PageArtifact]] = []

        async def extract(
            self,
            *,
            page_artifacts: list[PageArtifact],
            config: Config,
        ) -> EngineExtractionResult:
            self.calls.append(page_artifacts)
            page_num = page_artifacts[0].page_num
            return EngineExtractionResult(
                engine_id=self.engine_id,
                biomarkers=[
                    {
                        "raw_name": f"Marker {page_num}",
                        "value": float(page_num),
                        "unit": "mg/dL",
                        "source_engine": self.engine_id,
                    }
                ],
                notes=["shared note", f"page {page_num}"],
                metadata=ReportMetadata(
                    blood_collection={"date": "2025-01-01" if page_num == 1 else None},
                ),
                raw_payload=f'{{"page": {page_num}}}',
            )

    engine = PageEngine()

    result = asyncio.run(
        extract_report(
            image_paths=["/tmp/a.jpg", "/tmp/b.jpg"],
            config=Config(),
            engines=[engine],
        )
    )

    assert len(engine.calls) == 2
    assert [len(call) for call in engine.calls] == [1, 1]
    assert [call[0].page_num for call in engine.calls] == [1, 2]
    assert [biomarker.raw_name for biomarker in result.biomarkers] == [
        "Marker 1",
        "Marker 2",
    ]
    assert result.engine_results[0].biomarkers[0].page_num == 1
    assert result.engine_results[0].biomarkers[1].page_num == 2
    assert result.notes == ["shared note", "page 1", "page 2"]
    assert result.metadata.blood_collection.date == "2025-01-01"
    raw_payload = json.loads(result.raw_payload)
    assert raw_payload["engines"]["page_stub"]["notes"] == ["shared note", "page 1", "page 2"]


def test_config_resolved_extraction_engines_defaults_to_gemini():
    config = Config()
    specs = config.resolved_extraction_engines

    assert len(specs) == 1
    assert specs[0].type == "gemini_vision"
    assert specs[0].model == config.ocr


def test_config_keeps_explicit_extraction_engine_specs():
    config = Config(
        extraction_engines=[
            ExtractionEngineSpec(
                type="gemini_vision",
                id="gemini_page",
                execution_mode="page",
                model="gemini-2.5-flash",
            ),
            ExtractionEngineSpec(
                type="liteparse_text",
                id="liteparse_aux",
                execution_mode="document",
                cli_path="/tmp/liteparse",
            ),
        ]
    )

    specs = config.resolved_extraction_engines

    assert [spec.resolved_id() for spec in specs] == ["gemini_page", "liteparse_aux"]
    assert specs[0].execution_mode == "page"
