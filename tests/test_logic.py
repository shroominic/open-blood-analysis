import logging

import pytest

from app.logic import analyze_value, convert_units
from app.types import BiomarkerEntry, ReferenceRangeRule


def test_analyze_value_converts_units_with_normalized_lookup():
    entry = BiomarkerEntry(
        id="glucose",
        aliases=["Glucose"],
        canonical_unit="mmol/L",
        min_normal=3.9,
        max_normal=5.5,
        conversions={"mg/dL": "x / 18"},
    )

    result = analyze_value(
        raw_name="Glucose",
        raw_value=90.0,
        raw_unit="mg/dl",
        entry=entry,
    )

    assert result.unit == "mmol/L"
    assert result.status == "normal"
    assert result.value == pytest.approx(5.0)
    assert result.notes == "Converted from 90.0 mg/dl"


def test_analyze_value_does_not_mark_case_only_unit_changes_as_conversion():
    entry = BiomarkerEntry(
        id="glucose",
        aliases=["Glucose"],
        canonical_unit="mg/dL",
        min_normal=70,
        max_normal=99,
        conversions={},
    )

    result = analyze_value(
        raw_name="Glucose",
        raw_value=90.0,
        raw_unit="mg/dl",
        entry=entry,
    )

    assert result.status == "normal"
    assert result.notes is None


def test_analyze_value_qualitative_without_normal_values_is_unknown():
    entry = BiomarkerEntry(
        id="nitrite",
        aliases=["Nitrite"],
        canonical_unit="",
        min_normal=None,
        max_normal=None,
        conversions={},
        normal_values=None,
    )

    result = analyze_value(
        raw_name="Nitrite",
        raw_value="Positive",
        raw_unit="",
        entry=entry,
    )

    assert result.status == "unknown"


def test_analyze_value_logs_invalid_reference_rule_variables(caplog):
    entry = BiomarkerEntry(
        id="ldl_cholesterol",
        aliases=["LDL"],
        canonical_unit="mg/dL",
        min_normal=0,
        max_normal=100,
        conversions={},
        reference_rules=[
            ReferenceRangeRule(
                condition="risk_factor == 'diabetes'",
                min_normal=0,
                max_normal=70,
                priority=1,
            )
        ],
    )

    caplog.set_level(logging.WARNING, logger="app.logic")
    result = analyze_value(
        raw_name="LDL",
        raw_value=95.0,
        raw_unit="mg/dL",
        entry=entry,
        sex="male",
        age=35,
    )

    assert result.status == "normal"
    assert "unknown variable" in caplog.text.lower()
    assert "risk_factor" in caplog.text


def test_analyze_value_skips_age_rules_without_warning_when_age_missing(caplog):
    entry = BiomarkerEntry(
        id="test_biomarker",
        aliases=["Test"],
        canonical_unit="mg/dL",
        min_normal=0.0,
        max_normal=100.0,
        conversions={},
        reference_rules=[
            ReferenceRangeRule(
                condition="age < 18",
                min_normal=0.0,
                max_normal=80.0,
                priority=1,
            )
        ],
    )

    caplog.set_level(logging.WARNING, logger="app.logic")
    result = analyze_value(
        raw_name="Test",
        raw_value=90.0,
        raw_unit="mg/dL",
        entry=entry,
        age=None,
    )

    assert result.status == "normal"
    assert caplog.text == ""


def test_convert_units_supports_generic_mass_scaling_roundtrip():
    entry = BiomarkerEntry(
        id="apob",
        aliases=["ApoB"],
        canonical_unit="g/L",
        min_normal=0.0,
        max_normal=2.0,
        conversions={},
    )

    g_per_l, unit = convert_units(90.0, "mg/dL", "g/L", entry)
    assert unit == "g/L"
    assert g_per_l == pytest.approx(0.9)

    back, back_unit = convert_units(float(g_per_l), "g/L", "mg/dL", entry)
    assert back_unit == "mg/dL"
    assert back == pytest.approx(90.0)


def test_analyze_value_populates_reference_optimal_and_peak_metadata():
    entry = BiomarkerEntry(
        id="rdw",
        aliases=["RDW"],
        canonical_unit="%",
        min_normal=11.5,
        max_normal=14.5,
        min_optimal=11.8,
        max_optimal=12.8,
        peak_value=10.0,
        conversions={},
    )

    result = analyze_value(
        raw_name="RDW",
        raw_value=12.2,
        raw_unit="%",
        entry=entry,
    )

    assert result.status == "optimal"
    assert result.reference_status == "normal"
    assert result.optimal_status == "optimal"
    assert result.min_reference == 11.5
    assert result.max_reference == 14.5
    assert result.min_optimal == 11.8
    assert result.max_optimal == 12.8
    assert result.peak_value == 10.0


def test_analyze_value_hides_optimal_when_identical_to_reference():
    entry = BiomarkerEntry(
        id="x",
        aliases=["X"],
        canonical_unit="mg/dL",
        min_normal=10.0,
        max_normal=20.0,
        min_optimal=10.0,
        max_optimal=20.0,
        conversions={},
    )

    result = analyze_value(
        raw_name="X",
        raw_value=15.0,
        raw_unit="mg/dL",
        entry=entry,
    )

    assert result.status == "normal"
    assert result.reference_status == "normal"
    assert result.optimal_status == "not_applicable"
    assert result.min_reference == 10.0
    assert result.max_reference == 20.0
    assert result.min_optimal is None
    assert result.max_optimal is None


def test_analyze_value_marks_normal_but_low_optimal():
    entry = BiomarkerEntry(
        id="x",
        aliases=["X"],
        canonical_unit="mg/dL",
        min_normal=10.0,
        max_normal=20.0,
        min_optimal=13.0,
        max_optimal=18.0,
        conversions={},
    )

    result = analyze_value(
        raw_name="X",
        raw_value=11.0,
        raw_unit="mg/dL",
        entry=entry,
    )

    assert result.reference_status == "normal"
    assert result.optimal_status == "below_optimal"
    assert result.status == "moderate"


def test_analyze_value_marks_normal_but_high_optimal_as_elevated():
    entry = BiomarkerEntry(
        id="x",
        aliases=["X"],
        canonical_unit="mg/dL",
        min_normal=10.0,
        max_normal=20.0,
        min_optimal=12.0,
        max_optimal=16.0,
        conversions={},
    )

    result = analyze_value(
        raw_name="X",
        raw_value=18.0,
        raw_unit="mg/dL",
        entry=entry,
    )

    assert result.reference_status == "normal"
    assert result.optimal_status == "above_optimal"
    assert result.status == "elevated"


def test_convert_units_supports_mass_to_molar_with_molar_mass():
    entry = BiomarkerEntry(
        id="glucose",
        aliases=["Glucose"],
        canonical_unit="mmol/L",
        min_normal=3.9,
        max_normal=5.5,
        molar_mass_g_per_mol=180.156,
        conversions={},
    )

    mmol_l, unit = convert_units(90.0, "mg/dL", "mmol/L", entry)
    assert unit == "mmol/L"
    assert mmol_l == pytest.approx(4.9957, rel=1e-3)


def test_analyze_value_boolean_type_uses_typed_status():
    entry = BiomarkerEntry(
        id="nitrite",
        aliases=["Nitrite"],
        canonical_unit="",
        value_type="boolean",
        normal_values=["negative", "not detected"],
        min_normal=None,
        max_normal=None,
        conversions={},
    )

    normal = analyze_value(
        raw_name="Nitrite",
        raw_value="Negative",
        raw_unit="",
        entry=entry,
    )
    abnormal = analyze_value(
        raw_name="Nitrite",
        raw_value="Positive",
        raw_unit="",
        entry=entry,
    )

    assert normal.status == "normal"
    assert normal.value is False
    assert abnormal.status == "abnormal"
    assert abnormal.value is True


def test_analyze_value_enum_type_checks_allowed_and_normal_values():
    entry = BiomarkerEntry(
        id="urine_ketones",
        aliases=["Ketones"],
        canonical_unit="",
        value_type="enum",
        enum_values=["negative", "trace", "1+", "2+", "3+"],
        normal_values=["negative", "trace"],
        min_normal=None,
        max_normal=None,
        conversions={},
    )

    normal = analyze_value(
        raw_name="Ketones",
        raw_value="Trace",
        raw_unit="",
        entry=entry,
    )
    abnormal = analyze_value(
        raw_name="Ketones",
        raw_value="2+",
        raw_unit="",
        entry=entry,
    )
    unknown = analyze_value(
        raw_name="Ketones",
        raw_value="invalid-token",
        raw_unit="",
        entry=entry,
    )

    assert normal.status == "normal"
    assert abnormal.status == "abnormal"
    assert unknown.status == "unknown"
