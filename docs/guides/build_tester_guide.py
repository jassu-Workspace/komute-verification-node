"""Build the Postman testing guide for the testers' team (ASD-STE100 style).

    python docs/guides/build_tester_guide.py      # needs reportlab
Facts come from postman/, scripts/make_postman_collection.py, run.ps1, setup.ps1, setup.sh and the integration guide.
The only key printed is the shared TEST key for local test nodes, which run.ps1 -ApiKey accepts.
"""
from __future__ import annotations

from build_guides import C, P, PageBreak, Spacer, base, build, bullets, code, grid, h1, h2, h3, note, small, steps

TEST_KEY = "komute-local-TEST-key-not-for-production"

s = [P("Komute-Local: API testing with Postman", h1),
     P("Guide for the testers' team. Written in ASD-STE100 Simplified Technical English. 15 November 2026.", small),
     Spacer(1, 6)]

s += [P("1. Connection data and credentials", h2),
      grid([["Item", "Value"],
            ["Base URL (local laptop)", C("http://127.0.0.1:8000")],
            ["Postman environment variable", C("baseUrl")],
            ["Authentication header", C("X-API-Key")],
            ["Authentication: default", "OFF. Start with " + C(".\\run.ps1") + ". Send no key."],
            ["Authentication: test key", C(TEST_KEY) + "<br/>Start the node with " + C(".\\run.ps1 -ApiKey " + TEST_KEY)
             + ". Use the Postman environment \"Komute Local (auth on)\". It contains this key."],
            ["Wrong key (used by the tests)", C("wrong-key") + ". The node must answer 401."],
            ["Linux test server", "Base URL " + C("http://127.0.0.1:8000") + " on the server. The key is different on each "
             "server. Get it on the server with " + C("sudo grep ^KOMUTE_API_KEY= /etc/komute/komute.env")],
            ["Endpoints", C("POST /v1/verify") + ", " + C("GET /v1/sessions/{request_id}") + ", " + C("GET /health")
             + ", " + C("GET /v1/system") + ", " + C("GET /")]],
           [0.3, 0.7]),
      Spacer(1, 4),
      note("WARNING", "The test key is for local test nodes only. Everybody who has this document knows it. "
                      "Do not use it on a server that is open to a network, and never in production."),
      Spacer(1, 3),
      note("NOTE", "Each installation also makes its own random key in %LOCALAPPDATA%\\komute\\dev.env (Windows) or "
                   "/etc/komute/komute.env (Linux). Use that key with .\\run.ps1 -WithApiKey. The test key does not "
                   "replace it.")]

s += [P("2. Files in the postman folder", h2),
      grid([["File", "Contents"],
            [C("Komute-Local.postman_collection.json"), "33 requests in 7 folders. Each request has automatic tests."],
            [C("Komute-Local.postman_environment.json"), "Environment \"Komute Local\": authentication off"],
            [C("Komute-Local-auth-on.postman_environment.json"), "Environment \"Komute Local (auth on)\": the test key"],
            [C("samples/licence_back_fake.png"), "Back of a FAKE Ontario licence with a real AAMVA PDF417 barcode"],
            [C("samples/no_face.png"), "Plain image with no face. Used as selfie, licence front and unreadable barcode."],
            [C("samples/vehicle.png"), "Synthetic car photo"]],
           [0.45, 0.55]),
      Spacer(1, 3),
      note("WARNING", "All sample data is fake. Do not put real licence photos in the postman folder.")]

s += [P("3. Start the node (Windows laptop)", h2)]
s += steps([
    ("Install one time. Open PowerShell in the Komute-Local folder:",
     [code("powershell -ExecutionPolicy Bypass -File .\\setup.ps1")]),
    ("Start the node with authentication off:", [code("powershell -ExecutionPolicy Bypass -File .\\run.ps1 -NoBrowser")]),
    ("Or start the node with the test key:",
     [code("powershell -ExecutionPolicy Bypass -File .\\run.ps1 -NoBrowser -ApiKey " + TEST_KEY)]),
    f"Wait for the line {C('Running at http://127.0.0.1:8000')}. Keep the window open.",
    "To stop the node, push Ctrl+C.",
])

s += [P("4. Set up Postman", h2)]
s += steps([
    "Open Postman. Select Import (top left).",
    "Drag the three JSON files from the postman folder into the window. Select Import.",
    "In the environment menu (top right), select \"Komute Local\" (authentication off) or "
    "\"Komute Local (auth on)\" (test key).",
    "Select Settings (gear icon), General, Working directory, Choose. Select the postman folder, for example "
    + C("C:\\Komute-Local\\postman") + ". Postman needs this to find the sample images.",
])
s += [Spacer(1, 3),
      note("CAUTION", "The environment must agree with the node. Authentication off on the node: use \"Komute Local\". "
                      "-ApiKey on the node: use \"Komute Local (auth on)\". If they do not agree, many tests fail with 401.")]

s += [PageBreak(), P("5. Run the tests", h2)]
s += steps([
    "One request: open it and select Send. The Test Results tab shows each check.",
    "All requests: right-click the collection \"Komute-Local API\". Select Run collection, then Run Komute-Local API.",
    "Do not change the folder sequence. Folder 3 uses the request_id that folder 2 saves.",
])
s += [Spacer(1, 3),
      P("Correct result: 33 requests, 0 failures. One run takes approximately 10 seconds.", base),
      P("Command line (Node.js necessary):", h3),
      code("npx --yes newman@6 run postman/Komute-Local.postman_collection.json \\\n"
           "  -e postman/Komute-Local.postman_environment.json --working-dir postman"),
      P("With the test key, use the environment file Komute-Local-auth-on.postman_environment.json. Newman can stop "
        "after the summary when authentication is on. The results are then complete. Push Ctrl+C.", base)]

s += [P("6. What each folder tests", h2),
      grid([["Folder", "Requests", "Correct result"],
            ["1. Health and monitoring", "GET /health, GET /v1/system (JSON and browser), GET /",
             "Node ready, models loaded, memory under the limit, no secret in any answer"],
            ["2. Verify: main flow", "POST /v1/verify with the fake licence",
             "Stage 1 passes all 7 cross-checks. The selfie has no face, so: REJECTED with FACE_NOT_FOUND_SELFIE "
             "(retryable). The audit record is signed."],
            ["3. Idempotency", "Same request again; same request_id with changed data; GET /v1/sessions/{id}; unknown id",
             "Same stored result; 409; the stored session; 404"],
            ["4. Stage 1 cross-checks", "Wrong class, issue date, expiry date, date of birth, licence number, province, name",
             "Each gives its final reason code. Class case and licence hyphens are ignored. A one-letter name "
             "error scores approximately 94% and passes (90% pass mark)."],
            ["5. Barcode problems", "Licence back without a barcode",
             "BARCODE_UNREADABLE (retryable) with the retake message. Face and vehicle stages do not run."],
            ["6. Input validation", "Missing or incorrect fields", "422 with MISSING_FIELD, INVALID_FIELD or CONSENT_REQUIRED"],
            ["7. Authentication", "Wrong key, no key, verify without a key",
             "401 when authentication is on. When authentication is off, the key is ignored."]],
           [0.2, 0.37, 0.43]),
      Spacer(1, 3),
      P("Each request also checks the security headers and a response time below 60 seconds.", base)]

s += [P("7. Test data (collection variables)", h2),
      P("The claimed data agrees with the fake barcode. If you change one, change both.", base), Spacer(1, 3),
      grid([["Variable", "Value"],
            ["fullName", "Alex Jamie Tester"], ["dateOfBirth", "1990-03-15"], ["licenceNumber", "T12345678901234"],
            ["province / licenceClass", "ON / G"], ["issueDate / expiryDate", "2023-09-14 / 2030-12-31"],
            ["plateNumber / vehicleColour", "ABC123 / Grey"],
            ["driverId / consentAt / consentPolicyVersion", "pm-driver-1 / 2026-11-12T10:00:00Z / v1"],
            ["rid", "New request_id for each request (pre-request script)"]],
           [0.4, 0.6]),
      Spacer(1, 3),
      note("NOTE", "The fake licence expires on 2030-12-31. After that date, the result is LICENCE_EXPIRED.")]

s += [P("8. Responses to expect", h2),
      grid([["HTTP", "Meaning"],
            ["200", "A decision: APPROVED or REJECTED, with reason codes"],
            ["401", "UNAUTHORIZED: wrong or missing X-API-Key (authentication on)"],
            ["404", "Unknown request_id in GET /v1/sessions/{id}"],
            ["409", "REQUEST_ID_CONFLICT: same request_id with different data"],
            ["422", "Bad input: MISSING_FIELD, INVALID_FIELD, CONSENT_REQUIRED, INVALID_IMAGE, IMAGE_TOO_LARGE"],
            ["503", "BUSY: one verification is in progress. Wait Retry-After seconds."],
            ["500", "ENGINE_ERROR: internal fault. This is not a rejection."]],
           [0.12, 0.88])]

s += [P("9. Not in the collection", h2)]
s += bullets([
    "503 BUSY: send two verifications at the same time (for example from two Postman tabs).",
    "500 ENGINE_ERROR: needs an internal fault. The unit tests simulate it.",
    "An APPROVED decision: needs a real face that agrees with the licence photo. To test it, duplicate the request "
    "\"Verify: fake licence, barcode passes\" and select your own photos in the Body tab. Type the agreeing data. "
    "Keep those photos outside the code folder.",
])

s += [P("10. Problems", h2),
      grid([["Problem", "Action"],
            ["\"not in your working directory\"", "Do step 4.4 again (Working directory = the postman folder)"],
            ["Many 401 failures", "The environment does not agree with the node. See the CAUTION in section 4."],
            ["\"connect ECONNREFUSED\"", "The node is not running. Start it (section 3)."],
            ["LICENCE_EXPIRED in folder 2", "The date is after 2030-12-31. Ask the developer for new fake samples."]],
           [0.35, 0.65])]

build("komute-local-postman-testing.pdf", "Postman testing guide", s)
