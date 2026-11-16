"""Generate the Postman collection, environment and fake sample images. Run from the repo root:

    "$PY" scripts/make_postman_collection.py

Every value is fake (tests/fixtures/aamva_factory.py). Never point this at real licence data.
"""
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import backend.core  # noqa: E402,F401  (clamps before cv2)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import zxingcpp  # noqa: E402

from tests.fixtures.aamva_factory import make_aamva_text  # noqa: E402

OUT = pathlib.Path("postman")


def make_samples() -> None:
    d = OUT / "samples"
    d.mkdir(parents=True, exist_ok=True)
    bar = np.array(zxingcpp.write_barcode_to_image(
        zxingcpp.create_barcode(make_aamva_text().encode("latin-1"), zxingcpp.BarcodeFormat.PDF417), scale=3))
    card = np.full((bar.shape[0] + 160, bar.shape[1] + 160), 235, np.uint8)
    card[80:80 + bar.shape[0], 80:80 + bar.shape[1]] = bar
    cv2.imwrite(str(d / "licence_back_fake.png"), card)
    blank = np.full((480, 640, 3), 210, np.uint8)
    cv2.putText(blank, "NO FACE - TEST IMAGE", (90, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (90, 90, 90), 2)
    cv2.imwrite(str(d / "no_face.png"), blank)
    shutil.copyfile("tests/fixtures/synthetic_vehicle.png", d / "vehicle.png")


make_samples()

BACK, NOFACE, VEHICLE = "samples/licence_back_fake.png", "samples/no_face.png", "samples/vehicle.png"

# Fake card data: matches tests/fixtures/aamva_factory.py FAKE_FIELDS (never real data).
CLAIM = {
    "claimed.full_name": "{{fullName}}",
    "claimed.date_of_birth": "{{dateOfBirth}}",
    "claimed.licence_number": "{{licenceNumber}}",
    "claimed.issuing_province": "{{province}}",
    "claimed.licence_class": "{{licenceClass}}",
    "claimed.issue_date": "{{issueDate}}",
    "claimed.expiry_date": "{{expiryDate}}",
    "claimed.plate_number": "{{plateNumber}}",
    "claimed.vehicle_colour": "{{vehicleColour}}",
    "claimed.vehicle_make": "Toyota",
    "claimed.vehicle_model": "Corolla",
    "claimed.vehicle_year": "2020",
}
BASE_FIELDS = {
    "request_id": "{{rid}}",
    "driver_id": "{{driverId}}",
    "consent_at": "{{consentAt}}",
    "consent_policy_version": "{{consentPolicyVersion}}",
    **CLAIM,
}
FILES = {"selfie": NOFACE, "licence_front": NOFACE, "licence_back": BACK, "vehicle": VEHICLE}


def script(kind, lines):
    return {"listen": kind, "script": {"type": "text/javascript", "exec": lines}}


def tests(*lines):
    return [script("test", list(lines))]


def get(name, path, test_lines, headers=None, desc=""):
    return {
        "name": name,
        "event": tests(*test_lines),
        "request": {
            "method": "GET",
            "header": [{"key": k, "value": v} for k, v in (headers or {}).items()],
            "url": {"raw": "{{baseUrl}}" + path, "host": ["{{baseUrl}}"], "path": [p for p in path.split("/") if p]},
            "description": desc,
        },
    }


def verify(name, test_lines, override=None, drop=(), files=None, rid="{{rid}}", desc="", headers=None):
    fields = dict(BASE_FIELDS, request_id=rid)
    fields.update(override or {})
    body = [{"key": k, "value": v, "type": "text"} for k, v in fields.items() if k not in drop]
    for k, src in (files or FILES).items():
        if k not in drop:
            body.append({"key": k, "type": "file", "src": src})
    return {
        "name": name,
        "event": tests(*test_lines),
        "request": {
            "method": "POST",
            "header": [{"key": k, "value": v} for k, v in (headers or {}).items()],
            "body": {"mode": "formdata", "formdata": body},
            "url": {"raw": "{{baseUrl}}/v1/verify", "host": ["{{baseUrl}}"], "path": ["v1", "verify"]},
            "description": desc,
        },
    }


def folder(name, desc, items):
    return {"name": name, "description": desc, "item": items}


J = "const body = pm.response.json();"
STATUS200 = 'pm.test("HTTP 200", () => pm.response.to.have.status(200));'
REJECTED = 'pm.test("decision is REJECTED", () => pm.expect(body.decision).to.eql("REJECTED"));'


def codes_include(*codes):
    arr = json.dumps(list(codes))
    return (f'pm.test("reasons include {", ".join(codes)}", () => {{ const c = body.reasons.map(r => r.code); '
            f'{arr}.forEach(x => pm.expect(c).to.include(x)); }});')


def codes_exclude(*codes):
    arr = json.dumps(list(codes))
    return (f'pm.test("reasons exclude {", ".join(codes)}", () => {{ const c = body.reasons.map(r => r.code); '
            f'{arr}.forEach(x => pm.expect(c).to.not.include(x)); }});')


def check_row(field, passed):
    return (f'pm.test("check row {field} {"passes" if passed else "fails"}", () => {{ '
            f'const row = body.stages.barcode.checks.find(c => c.field === "{field}"); '
            f'pm.expect(row, "row present").to.be.an("object"); pm.expect(row.passed).to.eql({str(passed).lower()}); }});')


def error(status, code):
    return [J, f'pm.test("HTTP {status}", () => pm.response.to.have.status({status}));',
            f'pm.test("error code {code}", () => pm.expect(body.error.code).to.eql("{code}"));',
            'pm.test("no decision in an error", () => pm.expect(body).to.not.have.property("decision"));']


BARCODE_PASSES = [
    'pm.test("barcode stage ran and every licence rule passed", () => { const b = body.stages.barcode; '
    'pm.expect(b.ran && b.readable && b.aamva).to.be.true; pm.expect(b.jurisdiction).to.eql("ON"); pm.expect(b.gates_failed).to.eql([]); });',
    'pm.test("7 fields cross-checked, all matched", () => { const c = body.stages.barcode.checks; '
    'pm.expect(c).to.have.length(7); c.forEach(r => pm.expect(r.passed, r.field).to.be.true); });',
]

health = folder("1. Health and monitoring", "Read-only endpoints. They never change state.", [
    get("Health", "/health", [
        J,
        'pm.test("HTTP 200 (503 means a required model failed to load)", () => pm.response.to.have.status(200));',
        'pm.test("status ok", () => pm.expect(body.status).to.eql("ok"));',
        'pm.test("required models loaded", () => ["face_embedder", "plate_detector", "plate_ocr"].forEach(m => pm.expect(body.models_loaded[m], m).to.be.true));',
        'pm.test("rss_mb is a number", () => pm.expect(body.rss_mb).to.be.a("number"));',
    ], desc="Liveness probe for the host or load balancer. 200 = ready, 503 = degraded."),
    get("System figures (JSON)", "/v1/system", [
        J, STATUS200,
        'pm.test("has live sample, models and engine state", () => { pm.expect(body.sample.memory.process_rss_mb).to.be.above(0); '
        'pm.expect(body.models).to.be.an("array").that.is.not.empty; pm.expect(body.engine).to.have.property("busy"); });',
        'pm.test("never returns secrets", () => ["api_key", "storage_key_b64", "manifest_key_b64"].forEach(k => pm.expect(body.settings).to.not.have.property(k)));',
        'pm.test("memory under the 1200 MB limit", () => pm.expect(body.sample.memory.process_rss_mb).to.be.below(body.budget_mb.peak_limit));',
    ], headers={"Accept": "application/json"}, desc="Data behind the /stats dashboard. Needs X-API-Key when authentication is on."),
    get("System dashboard (browser view)", "/v1/system", [
        STATUS200,
        'pm.test("returns the HTML dashboard", () => { pm.expect(pm.response.headers.get("Content-Type")).to.include("text/html"); pm.expect(pm.response.text()).to.include("Komute System Monitor"); });',
    ], headers={"Accept": "text/html"}, desc="A browser gets the dashboard page; open {{baseUrl}}/v1/system in a browser to see it live."),
    get("Test page", "/", [STATUS200, 'pm.test("HTML test page", () => pm.expect(pm.response.text()).to.include("Komute Verify"));']),
])

verify_folder = folder("2. Verify: main flow", (
    "POST /v1/verify with the fake licence. The barcode stage passes; the selfie has no face, so the result is "
    "REJECTED with the retryable FACE_NOT_FOUND_SELFIE. That proves Stage 1 end to end without real personal data."
), [
    verify("Verify: fake licence, barcode passes", [
        J, STATUS200, REJECTED, *BARCODE_PASSES,
        codes_include("FACE_NOT_FOUND_SELFIE"),
        'pm.test("FACE_NOT_FOUND_SELFIE is retryable", () => pm.expect(body.reasons.find(r => r.code === "FACE_NOT_FOUND_SELFIE").retryable).to.be.true);',
        'pm.test("face stage ran", () => pm.expect(body.stages.face.ran).to.be.true);',
        'pm.test("audit manifest is signed", () => { pm.expect(body.audit.manifest_id).to.be.a("string").and.not.empty; pm.expect(body.audit.key_id).to.be.a("string"); });',
        'pm.test("timings and memory reported", () => { pm.expect(body.timings_ms.total).to.be.above(0); pm.expect(body.rss_mb).to.be.above(0); });',
        'pm.test("full barcode fields only if the node enables them", () => { if (body.barcode_fields) pm.expect(body.barcode_fields.DAQ).to.eql("T12345678901234"); });',
        '// keep this request for the idempotency folder',
        'pm.collectionVariables.set("savedRid", pm.collectionVariables.get("rid"));',
        'pm.collectionVariables.set("savedManifest", body.audit.manifest_id);',
        'pm.collectionVariables.set("savedDecision", body.decision);',
    ], desc="Main happy path for Stage 1. Saves its request_id for the next folder."),
])

idem = folder("3. Idempotency and stored results", "Uses the request_id saved by folder 2. Run folder 2 first.", [
    verify("Replay: same request_id, same data", [
        J, STATUS200,
        'pm.test("returns the stored result (same manifest)", () => pm.expect(body.audit.manifest_id).to.eql(pm.collectionVariables.get("savedManifest")));',
        'pm.test("same decision", () => pm.expect(body.decision).to.eql(pm.collectionVariables.get("savedDecision")));',
    ], rid="{{savedRid}}", desc="The node does not re-run a verification it already decided."),
    verify("Replay: same request_id, different data", error(409, "REQUEST_ID_CONFLICT"),
           override={"claimed.plate_number": "ZZZ999"}, rid="{{savedRid}}"),
    get("Get stored session", "/v1/sessions/{{savedRid}}", [
        J, STATUS200,
        'pm.test("matches the original", () => { pm.expect(body.request_id).to.eql(pm.collectionVariables.get("savedRid")); pm.expect(body.audit.manifest_id).to.eql(pm.collectionVariables.get("savedManifest")); });',
    ]),
    get("Get unknown session", "/v1/sessions/pm-does-not-exist", error(404, "NOT_FOUND")),
])

cross = folder("4. Stage 1 cross-checks", (
    "Each request changes one claimed field. A mismatch is a final reason code; Stages 2 and 3 do not run."
), [
    verify("Wrong licence class", [J, STATUS200, REJECTED, codes_include("CLASS_MISMATCH"), check_row("licence_class", False),
                                   'pm.test("CLASS_MISMATCH is final", () => pm.expect(body.reasons.find(r => r.code === "CLASS_MISMATCH").retryable).to.be.false);',
                                   'pm.test("face stage skipped", () => pm.expect(body.stages.face.ran).to.be.false);'],
           override={"claimed.licence_class": "M"}),
    verify("Licence class: case and spaces ignored", [J, STATUS200, *BARCODE_PASSES], override={"claimed.licence_class": " g "}),
    verify("Wrong issue date", [J, STATUS200, REJECTED, codes_include("ISSUE_DATE_MISMATCH"), check_row("issue_date", False)],
           override={"claimed.issue_date": "2023-09-15"}),
    verify("Wrong expiry date", [J, STATUS200, REJECTED, codes_include("EXPIRY_DATE_MISMATCH"), check_row("expiry_date", False)],
           override={"claimed.expiry_date": "2031-12-31"}),
    verify("Wrong date of birth", [J, STATUS200, REJECTED, codes_include("DOB_MISMATCH"), check_row("date_of_birth", False)],
           override={"claimed.date_of_birth": "1990-03-16"}),
    verify("Wrong licence number", [J, STATUS200, REJECTED, codes_include("LICENCE_NUMBER_MISMATCH"), check_row("licence_number", False)],
           override={"claimed.licence_number": "T99999999999999"}),
    verify("Licence number with hyphens matches", [J, STATUS200, *BARCODE_PASSES], override={"claimed.licence_number": "T1234-56789-01234"}),
    verify("Wrong province", [J, STATUS200, REJECTED, codes_include("JURISDICTION_MISMATCH"), check_row("issuing_province", False)],
           override={"claimed.issuing_province": "QC"}),
    verify("Name with one typo passes (about 94%)", [
        J, STATUS200, *BARCODE_PASSES, codes_exclude("NAME_MISMATCH"),
        'pm.test("name score at or above the 90% pass mark", () => { const r = body.stages.barcode.checks.find(c => c.field === "full_name"); pm.expect(r.score).to.be.at.least(r.threshold).and.below(100); });',
    ], override={"claimed.full_name": "Alex Jamie Testor"}),
    verify("Different name fails", [
        J, STATUS200, REJECTED, codes_include("NAME_MISMATCH"), check_row("full_name", False),
        'pm.test("name score below the pass mark", () => { const r = body.stages.barcode.checks.find(c => c.field === "full_name"); pm.expect(r.score).to.be.below(r.threshold); });',
    ], override={"claimed.full_name": "Alan Jamie Tester"}),
    verify("Several mismatches are all listed", [
        J, STATUS200, REJECTED, codes_include("CLASS_MISMATCH", "EXPIRY_DATE_MISMATCH", "DOB_MISMATCH"),
    ], override={"claimed.licence_class": "M", "claimed.expiry_date": "2031-12-31", "claimed.date_of_birth": "1991-03-15"}),
])

barcode = folder("5. Barcode problems", "What the driver sees when the barcode photo is bad.", [
    verify("Unreadable barcode", [
        J, STATUS200, REJECTED, codes_include("BARCODE_UNREADABLE"),
        'pm.test("retryable, with the retake message", () => { const r = body.reasons.find(x => x.code === "BARCODE_UNREADABLE"); pm.expect(r.retryable).to.be.true; pm.expect(r.message).to.include("not clearly visible"); });',
        'pm.test("later stages skipped", () => { pm.expect(body.stages.face.ran).to.be.false; pm.expect(body.stages.vehicle.ran).to.be.false; });',
    ], files=dict(FILES, licence_back=NOFACE), desc="The licence back is a plain image, so no PDF417 is found."),
])

bad = folder("6. Input validation (422)", "Bad requests return 422 with a specific code and never a decision.", [
    verify("Missing licence class", error(422, "MISSING_FIELD"), drop=("claimed.licence_class",)),
    verify("Missing expiry date", error(422, "MISSING_FIELD"), drop=("claimed.expiry_date",)),
    verify("Issue date in wrong format", error(422, "INVALID_FIELD"), override={"claimed.issue_date": "14/09/2023"}),
    verify("Unknown province", error(422, "INVALID_FIELD"), override={"claimed.issuing_province": "ZZ"}),
    verify("Missing consent", error(422, "CONSENT_REQUIRED"), drop=("consent_at",)),
    verify("Consent time without time zone", error(422, "INVALID_FIELD"), override={"consent_at": "2026-11-12T10:00:00"}),
    verify("Bad request_id characters", error(422, "INVALID_FIELD"), rid="bad id!"),
    verify("Missing selfie image", error(422, "MISSING_FIELD"), drop=("selfie",)),
    verify("Vehicle year not a year", error(422, "INVALID_FIELD"), override={"claimed.vehicle_year": "abc"}),
])

auth = folder("7. Authentication", (
    "Authentication is OFF by default. Start the node with .\\run.ps1 -WithApiKey, put the key from "
    "%LOCALAPPDATA%\\komute\\dev.env in the environment's apiKey, and set authEnabled to true."
), [
    get("Wrong API key", "/v1/system", [
        'if (pm.environment.get("authEnabled") === "true") {',
        '  pm.test("HTTP 401", () => pm.response.to.have.status(401));',
        '  pm.test("error code UNAUTHORIZED", () => pm.expect(pm.response.json().error.code).to.eql("UNAUTHORIZED"));',
        '} else {',
        '  pm.test("auth is off: the key is ignored (HTTP 200)", () => pm.response.to.have.status(200));',
        '}',
    ], headers={"X-API-Key": "wrong-key", "Accept": "application/json"}),
    get("No API key", "/v1/system", [
        'if (pm.environment.get("authEnabled") === "true") {',
        '  pm.test("HTTP 401", () => pm.response.to.have.status(401));',
        '} else {',
        '  pm.test("auth is off: no key needed (HTTP 200)", () => pm.response.to.have.status(200));',
        '}',
    ], headers={"Accept": "application/json"}),
    verify("Verify with no API key", [
        'if (pm.environment.get("authEnabled") === "true") {',
        '  pm.test("HTTP 401, nothing verified", () => pm.response.to.have.status(401));',
        '} else {',
        '  pm.test("auth is off: verification runs (HTTP 200)", () => pm.response.to.have.status(200));',
        '}',
    ]),
])
for item in auth["item"]:
    item["request"]["auth"] = {"type": "noauth"}  # do not inherit the collection key, or the wrong/missing key is replaced

collection = {
    "info": {
        "name": "Komute-Local API",
        "description": (
            "Driver and vehicle verification node. Read postman/README.md first.\n\n"
            "Set the Postman working directory to the repo's postman folder so the sample images load. "
            "All sample data is fake."
        ),
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
    },
    "auth": {"type": "apikey", "apikey": [
        {"key": "key", "value": "X-API-Key", "type": "string"},
        {"key": "value", "value": "{{apiKey}}", "type": "string"},
        {"key": "in", "value": "header", "type": "string"},
    ]},
    "event": [
        script("prerequest", [
            "// A fresh request_id for every request; replay requests use {{savedRid}} instead.",
            'pm.collectionVariables.set("rid", "pm-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6));',
        ]),
        script("test", [
            'pm.test("answers within 60 s", () => pm.expect(pm.response.responseTime).to.be.below(60000));',
            'pm.test("security headers present", () => { pm.expect(pm.response.headers.get("X-Content-Type-Options")).to.eql("nosniff"); '
            'pm.expect(pm.response.headers.get("Content-Security-Policy")).to.include("frame-ancestors \'none\'"); });',
        ]),
    ],
    "variable": [
        {"key": "fullName", "value": "Alex Jamie Tester"},
        {"key": "dateOfBirth", "value": "1990-03-15"},
        {"key": "licenceNumber", "value": "T12345678901234"},
        {"key": "province", "value": "ON"},
        {"key": "licenceClass", "value": "G"},
        {"key": "issueDate", "value": "2023-09-14"},
        {"key": "expiryDate", "value": "2030-12-31"},
        {"key": "plateNumber", "value": "ABC123"},
        {"key": "vehicleColour", "value": "Grey"},
        {"key": "driverId", "value": "pm-driver-1"},
        {"key": "consentAt", "value": "2026-11-12T10:00:00Z"},
        {"key": "consentPolicyVersion", "value": "v1"},
        {"key": "rid", "value": ""},
        {"key": "savedRid", "value": ""},
        {"key": "savedManifest", "value": ""},
        {"key": "savedDecision", "value": ""},
    ],
    "item": [health, verify_folder, idem, cross, barcode, bad, auth],
}

environment = {
    "name": "Komute Local",
    "values": [
        {"key": "baseUrl", "value": "http://127.0.0.1:8000", "type": "default", "enabled": True},
        {"key": "apiKey", "value": "", "type": "secret", "enabled": True},
        {"key": "authEnabled", "value": "false", "type": "default", "enabled": True},
    ],
    "_postman_variable_scope": "environment",
}

# Shared key for local test nodes only (start the node with: .\run.ps1 -ApiKey <this key>). Never use it in production.
TEST_API_KEY = "komute-local-TEST-key-not-for-production"
environment_auth = {
    "name": "Komute Local (auth on)",
    "values": [
        {"key": "baseUrl", "value": "http://127.0.0.1:8000", "type": "default", "enabled": True},
        {"key": "apiKey", "value": TEST_API_KEY, "type": "secret", "enabled": True},
        {"key": "authEnabled", "value": "true", "type": "default", "enabled": True},
    ],
    "_postman_variable_scope": "environment",
}

(OUT / "Komute-Local.postman_collection.json").write_text(json.dumps(collection, indent=2) + "\n", encoding="utf-8")
(OUT / "Komute-Local.postman_environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
(OUT / "Komute-Local-auth-on.postman_environment.json").write_text(json.dumps(environment_auth, indent=2) + "\n",
                                                                   encoding="utf-8")
n = sum(len(f["item"]) for f in collection["item"])
print("requests:", n)
