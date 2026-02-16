import json

from app.llm import _parse_llm_response


def test_parse_llm_response_coerces_numeric_like_strings():
    content = json.dumps(
        {
            "data": [
                {"raw_name": "Glucose", "value": "150", "unit": "mg/dL", "flags": []},
                {"raw_name": "TSH", "value": "1,54", "unit": "uIU/mL", "flags": []},
                {"raw_name": "CRP", "value": "< 5", "unit": "mg/L", "flags": ["<"]},
                {"raw_name": "Occult Blood", "value": True, "unit": "", "flags": []},
                {
                    "raw_name": "Nitrite",
                    "value": "Negative",
                    "unit": "",
                    "flags": [],
                },
            ],
            "notes": "Sample note",
            "metadata": {
                "patient": {"age": 44, "gender": "male"},
                "lab": {
                    "company_name": "MayerLab",
                    "location": "Asuncion",
                },
                "blood_collection": {
                    "date": "2026-01-05",
                    "time": "08:30",
                    "datetime": None,
                },
            },
        }
    )

    biomarkers, notes, metadata = _parse_llm_response(content)

    assert notes == ["Sample note"]
    assert len(biomarkers) == 5
    assert biomarkers[0].value == 150.0
    assert biomarkers[1].value == 1.54
    assert biomarkers[2].value == 5.0
    assert biomarkers[3].value is True
    assert biomarkers[4].value == "Negative"
    assert metadata.patient.age == 44
    assert metadata.patient.gender == "male"
    assert metadata.lab.company_name == "MayerLab"
    assert metadata.lab.location == "Asuncion"
    assert metadata.blood_collection.date == "2026-01-05"
    assert metadata.blood_collection.time == "08:30"
    assert metadata.blood_collection.datetime is None


def test_parse_llm_response_metadata_falls_back_to_flat_keys():
    content = json.dumps(
        {
            "data": [],
            "patient_age": "37 years",
            "patient_sex": "female",
            "lab_name": "Central Lab",
            "location": "Madrid",
            "collection_date": "2025-12-01",
            "collection_time": "07:45",
            "collection_datetime": "2025-12-01T07:45:00",
        }
    )

    biomarkers, notes, metadata = _parse_llm_response(content)

    assert biomarkers == []
    assert notes == []
    assert metadata.patient.age == 37
    assert metadata.patient.gender == "female"
    assert metadata.lab.company_name == "Central Lab"
    assert metadata.lab.location == "Madrid"
    assert metadata.blood_collection.date == "2025-12-01"
    assert metadata.blood_collection.time == "07:45"
    assert metadata.blood_collection.datetime == "2025-12-01T07:45:00"
