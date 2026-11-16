"""Build the two setup guides (ASD-STE100 style) as PDF.

    python docs/guides/build_guides.py        # needs reportlab
Writes docs/guides/komute-local-windows-setup.pdf and docs/guides/komute-local-linux-deploy-and-integration.pdf.
Facts come from setup.ps1, run.ps1, setup.sh, start.sh, backend/core/config.py and docs/INTEGRATION_GUIDE.md.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent
DATE = "15 November 2026"

INK, MUTED, LINE, ACCENT = (colors.HexColor(c) for c in ("#1b1f24", "#5b6169", "#d9d5cb", "#27407a"))
SOFT, CODEBG, WARNBG, WARNC = (colors.HexColor(c) for c in ("#f4f2ed", "#f1f3f6", "#fdf3e7", "#a15c07"))
base = ParagraphStyle("b", fontName="Helvetica", fontSize=9, leading=12, textColor=INK)
small = ParagraphStyle("s", parent=base, fontSize=7.8, leading=10, textColor=MUTED)
h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=17, leading=21)
h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=ACCENT,
                    spaceBefore=10, spaceAfter=4, keepWithNext=1)
h3 = ParagraphStyle("h3", parent=base, fontName="Helvetica-Bold", fontSize=9.4, leading=12, spaceBefore=6, spaceAfter=2,
                    keepWithNext=1)
cell = ParagraphStyle("c", parent=base, fontSize=8.3, leading=10.6)
cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold")
code_s = ParagraphStyle("code", parent=base, fontName="Courier", fontSize=8, leading=10.2)
W = A4[0] - 32 * mm


def P(text: str, style: ParagraphStyle = base) -> Paragraph:
    return Paragraph(text, style)


def C(text: str) -> str:
    """Inline code."""
    return f"<font name='Courier'>{escape(text)}</font>"


def code(text: str) -> Table:
    """A code block: escaped, line breaks kept, leading spaces kept."""
    lines = []
    for ln in text.strip("\n").split("\n"):
        stripped = ln.lstrip(" ")
        lines.append("&nbsp;" * (len(ln) - len(stripped)) + escape(stripped))
    t = Table([[Paragraph("<br/>".join(lines), code_s)]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CODEBG), ("LEFTPADDING", (0, 0), (-1, -1), 6),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def note(kind: str, text: str) -> Table:
    """ASD-STE100 safety notes: WARNING (risk to data or security), CAUTION (risk to the installation), NOTE."""
    t = Table([[P(f"<b>{kind}</b>", cellb), P(text, cell)]], colWidths=[W * 0.14, W * 0.86])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), WARNBG if kind != "NOTE" else SOFT),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.5, WARNC if kind != "NOTE" else ACCENT),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def grid(rows: list[list], widths: list[float], head: bool = True) -> Table:
    data = [[P(c, cellb if head and i == 0 else cell) if isinstance(c, str) else c for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[W * w for w in widths], repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
          ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
    if head:
        st += [("BACKGROUND", (0, 0), (-1, 0), SOFT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    t.setStyle(TableStyle(st))
    return t


def steps(items: list, start: int = 1) -> list:
    """Numbered procedure. An item is text, or (text, [flowables shown under the step])."""
    out = []
    for n, it in enumerate(items, start):
        text, extra = (it, []) if isinstance(it, str) else it
        row = Table([[P(f"<b>{n}.</b>", cell), P(text, base)]], colWidths=[W * 0.05, W * 0.95])
        row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
        out.append(KeepTogether([row] + [Spacer(1, 2)] + extra) if extra else row)
        if extra:
            out.append(Spacer(1, 3))
    return out


def bullets(items: list[str]) -> list:
    return [P(f"&bull;&nbsp;&nbsp;{t}", base) for t in items]


def build(name: str, title: str, story: list) -> None:
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 8 * mm, f"Komute-Local  |  {title}  |  {DATE}")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    path = OUT / name
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=15 * mm, title=title, author="Komute-Local")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(path)


# =========================================================================================== Windows guide
def windows() -> list:
    s = [P("Komute-Local: set up on a Windows laptop", h1),
         P(f"Procedure guide. Written in ASD-STE100 Simplified Technical English. {DATE}.", small), Spacer(1, 6)]

    s += [P("1. Purpose", h2),
          P("This guide gives the steps to install and start Komute-Local on one Windows laptop. "
            "Komute-Local is the verification node. It checks a driver licence, a selfie and a vehicle photo. "
            "It gives one decision: APPROVED or REJECTED. Use the laptop setup for development and tests.")]

    s += [P("2. Requirements", h2),
          grid([["Item", "Requirement"],
                ["Computer", "x64 (AMD64) processor"],
                ["Operating system", "Windows 10 version 1809 or later, or Windows Server 2019 or later"],
                ["PowerShell", "Version 5.1 or later (included in Windows 10 and 11)"],
                ["Memory (RAM)", "2 GB or more. One verification can use up to approximately 1.2 GB."],
                ["Free disk", f"3 GB or more on the drive of {C('%LOCALAPPDATA%')} (normally C:)"],
                ["Network", "Access to github.com, pypi.org and files.pythonhosted.org during setup"],
                ["Rights", "Administrator approval one time, if the Microsoft Visual C++ runtime is missing"]],
               [0.25, 0.75])]

    s += [P("3. Get the code", h2)]
    s += steps([
        ("On the development laptop, open a terminal in the Komute-Local folder. Make a ZIP file of the "
         "committed code:", [code("git archive --format=zip -o komute-local.zip HEAD")]),
        "Copy komute-local.zip to the new laptop.",
        f"Extract the ZIP file to a folder, for example {C('C:\\Komute-Local')}.",
    ])
    s += [Spacer(1, 3),
          note("WARNING", "Do not copy the folder \"Original Lisence's\" to other computers. It contains real licence "
                          "photos (personal data). The ZIP file from git archive does not contain this folder."),
          Spacer(1, 3),
          note("NOTE", "git archive includes only committed files. Commit all changes before you make the ZIP file.")]

    s += [P("4. Run the setup script", h2)]
    s += steps([
        "Open Windows PowerShell. You do not need \"Run as administrator\".",
        ("Go to the code folder:", [code("cd C:\\Komute-Local")]),
        ("Run the setup script:", [code("powershell -ExecutionPolicy Bypass -File .\\setup.ps1")]),
        "If Windows shows an administrator prompt for the Visual C++ runtime, select Yes.",
        "Wait for the script to finish. Do not close the window.",
        f"Make sure that the last line is: {C('Setup complete. Start the app with: .\\run.ps1')}",
    ])
    s += [Spacer(1, 3), P("The setup script does these tasks in sequence:", base)]
    s += bullets([
        "It checks the processor, Windows version, PowerShell version, RAM, free disk and network.",
        "It installs the Microsoft Visual C++ runtime if it is missing. It checks the Microsoft signature first.",
        "It installs uv (the Python package tool) with winget or with the official installer.",
        f"It installs Python 3.12 and makes the virtual environment {C('%LOCALAPPDATA%\\komute\\venv')}.",
        f"It installs the exact package versions from {C('requirements.lock')}.",
        f"It downloads the four model files. It deletes a file if its SHA-256 does not agree with {C('models.lock')}.",
        f"It makes development keys one time in {C('%LOCALAPPDATA%\\komute\\dev.env')}. It does not show the values.",
        "It checks the model hashes and the settings. Then it runs the unit tests (approximately 30 to 60 seconds).",
    ])
    s += [Spacer(1, 3),
          grid([["Option", "Result"],
                [C(".\\setup.ps1 -SkipTests"), "Do all steps, but do not run the unit tests"],
                [C(".\\setup.ps1 -Force"), "Download all models again and install all packages again"]],
               [0.35, 0.65]),
          Spacer(1, 3),
          note("CAUTION", "If the script says that Windows wants a restart, restart the laptop before the first real run.")]

    s += [P("5. Start the application", h2)]
    s += steps([
        ("In the same folder, run the start script:", [code("powershell -ExecutionPolicy Bypass -File .\\run.ps1")]),
        f"Wait for the line {C('Running at http://127.0.0.1:8000')}. The browser opens the test page.",
        "To stop the application, push Ctrl+C in the PowerShell window.",
    ])
    s += [Spacer(1, 3),
          P("The start script checks Python 3.12, the packages and the model hashes before it starts. "
            "If a package is missing, it installs it. If a model file is missing or changed, it stops. "
            "Then run the setup script again.", base),
          Spacer(1, 3),
          grid([["Option", "Result"],
                [C(".\\run.ps1 -Port 8010"), "Use port 8010 (default: 8000)"],
                [C(".\\run.ps1 -NoBrowser"), "Do not open the browser"],
                [C(".\\run.ps1 -WithApiKey"), "Make the X-API-Key header necessary. The key is in dev.env."],
                [C(".\\run.ps1 -CheckOnly"), "Do the checks only. Do not start the server."]],
               [0.35, 0.65]),
          Spacer(1, 3),
          note("NOTE", "Authentication is off by default on the laptop. This is for tests only. "
                       "No liveness model is installed, so the application skips the liveness check. "
                       "Each response then contains the advisory LIVENESS_SKIPPED_DEV.")]

    s += [P("6. Do a check of the installation", h2),
          grid([["Open this address", "Correct result"],
                [C("http://127.0.0.1:8000/"), "The test page opens. You can send a test verification."],
                [C("http://127.0.0.1:8000/health"), "The text contains \"status\":\"ok\""],
                [C("http://127.0.0.1:8000/v1/system"), "The system page shows CPU, memory and model status"]],
               [0.4, 0.6]),
          Spacer(1, 3),
          P(f"For API tests with Postman, see {C('postman/README.md')} in the code folder.", base)]

    s += [P("7. Files and folders", h2),
          grid([["Path", "Contents"],
                [C("%LOCALAPPDATA%\\komute\\venv"), "Python 3.12 and the packages"],
                [C("%LOCALAPPDATA%\\komute\\dev.env"), "API key, storage key and manifest key. Keep this file secret."],
                [C("%LOCALAPPDATA%\\komute\\storage"), "Encrypted session records and images"],
                [C("backend\\models"), "The four model files and models.lock"],
                [C(".env"), "Optional settings (copy .env.example to .env). Not in git."]],
               [0.42, 0.58]),
          Spacer(1, 3),
          note("CAUTION", "The storage folder must not be in OneDrive. The application stops if it is."),
          P("To change a setting, copy .env.example to .env. Edit the value. Then restart the application. "
            f"Example: {C('KOMUTE_FACE_MATCH_THRESHOLD=0.35')} is the face match pass mark.", base)]

    s += [P("8. Troubleshooting", h2),
          grid([["Problem", "Action"],
                ["\"running scripts is disabled on this system\"",
                 f"Use the full command: {C('powershell -ExecutionPolicy Bypass -File .\\setup.ps1')}"],
                ["\"Port 8000 is used by another program\"", f"Stop that program, or use {C('.\\run.ps1 -Port 8010')}"],
                ["A model file does not agree with models.lock", f"Run {C('.\\setup.ps1 -Force')}"],
                ["\"Cannot reach github.com\"", "Connect to the internet, or set HTTPS_PROXY. Then run the script again."],
                ["Tests failed", "Read the output. Run the setup script again. If it fails again, send the output to the developer."],
                ["The server did not become healthy", "Read the messages above the error. Run .\\run.ps1 -CheckOnly."]],
               [0.4, 0.6])]

    s += [P("9. Update to a new version", h2)]
    s += steps(["Make a new ZIP file (section 3) and extract it over the old folder.",
                "Run the setup script again. It keeps the keys in dev.env and the stored data.",
                "Start the application with the start script."])
    return s


# =========================================================================================== Linux guide
def linux() -> list:
    s = [P("Komute-Local: deploy on a Linux server and connect the main application", h1),
         P(f"Procedure and interface guide. Written in ASD-STE100 Simplified Technical English. {DATE}.", small),
         Spacer(1, 6)]

    s += [P("1. Purpose", h2),
          P("This guide gives the steps to install Komute-Local on an Ubuntu server. "
            "It also tells the main Komute application team how to connect to it. "
            "Komute-Local is a verification node. The main application sends it the claimed driver data and four photos. "
            "The node gives one decision: APPROVED or REJECTED.")]

    s += [P("2. Server requirements", h2),
          grid([["Item", "Requirement"],
                ["Operating system", "Ubuntu 22.04 or 24.04 LTS"],
                ["Processor", "x86_64. 2 vCPU is recommended. 1 vCPU operates, but each verification is slower."],
                ["Memory (RAM)", "2 GB or more. The setup script stops on a server with less memory. "
                                 "The service is limited to 1300 MB."],
                ["Free disk", "3 GB or more in /opt"],
                ["Network", "Access to github.com, astral.sh, pypi.org and files.pythonhosted.org during setup"],
                ["Access", "An SSH account with sudo rights"]],
               [0.25, 0.75]),
          Spacer(1, 4),
          note("WARNING", "The Ubuntu server report of 11 November 2026 shows 1 CPU core, 961 MiB RAM and 491 MiB "
                          "available RAM. The main application (Django, Daphne, Redis) also runs on it. "
                          "This server cannot run Komute-Local. Resize it before you start. "
                          "Recommended: 4 GB RAM and 2 vCPU if the node shares the server with the main application. "
                          "2 GB RAM is sufficient on a server for the node only."),
          Spacer(1, 3),
          note("NOTE", "Memory use was measured on Windows (approximately 330 MB steady, 380 MB peak). "
                       "It is not yet measured on Linux. Monitor it after the first deployment.")]

    s += [P("3. Make the deployment package (on the development laptop)", h2)]
    s += steps([
        "Commit all changes in git. The package contains committed files only.",
        ("Open Git Bash or PowerShell in the Komute-Local folder. Make the package:",
         [code("git archive --format=tar.gz -o komute-local.tar.gz HEAD")]),
        ("Copy the package to the server. Replace USER and SERVER_IP with your values:",
         [code("scp komute-local.tar.gz USER@SERVER_IP:/tmp/")]),
    ])
    s += [Spacer(1, 3),
          note("WARNING", "Do not copy the folder \"Original Lisence's\" to the server. It contains real licence photos. "
                          "git archive does not include it, because git ignores it.")]

    s += [P("4. Run the setup script (on the server)", h2)]
    s += steps([
        ("Connect to the server:", [code("ssh USER@SERVER_IP")]),
        ("Extract the package to a new folder:", [code("rm -rf /tmp/komute-src && mkdir /tmp/komute-src\n"
                                                       "tar -xzf /tmp/komute-local.tar.gz -C /tmp/komute-src\n"
                                                       "cd /tmp/komute-src")]),
        ("Run the setup script as root:", [code("sudo bash ./setup.sh")]),
        "Wait for the script to finish.",
        f"Make sure that you see {C('ok  node healthy at http://127.0.0.1:8000')} and {C('Setup complete.')}",
    ])
    s += [Spacer(1, 3), P("The setup script does these tasks in sequence:", base)]
    s += bullets([
        "It checks for root, Ubuntu, x86_64, 2 GB RAM, 3 GB free disk and network access.",
        "It installs the OS packages curl, rsync and ca-certificates.",
        f"It makes the system user {C('komute')}. The service runs as this user, not as root.",
        f"It copies the code to {C('/opt/komute/app')}. It keeps model files that are already there.",
        f"It installs uv, Python 3.12 ({C('/opt/komute/python')}) and the virtual environment ({C('/opt/komute/venv')}).",
        f"It installs the exact package versions from {C('requirements.lock')}.",
        f"It downloads the four model files. It deletes a file if its SHA-256 does not agree with {C('models.lock')}.",
        f"It makes the settings file {C('/etc/komute/komute.env')} one time, with new random keys. "
        "It does not show the key values.",
        f"It installs the systemd service {C('komute')}. The service starts at boot and runs {C('start.sh')}.",
        f"It starts the service and waits for {C('GET /health')} to give status 200.",
    ])
    s += [Spacer(1, 3),
          P(f"To install without a start, use {C('sudo bash ./setup.sh --no-start')}. "
            "You can run the script again at any time. It keeps the settings file, the keys and the stored data.", base)]

    s += [P("5. Operate the service", h2),
          grid([["Task", "Command"],
                ["Show the status", C("sudo systemctl status komute")],
                ["Start / stop / restart", C("sudo systemctl start komute") + "<br/>" + C("sudo systemctl stop komute")
                 + "<br/>" + C("sudo systemctl restart komute")],
                ["Show the log (live)", C("journalctl -u komute -f")],
                ["Do a health check", C("curl http://127.0.0.1:8000/health")],
                ["Show the API key", C("sudo grep ^KOMUTE_API_KEY= /etc/komute/komute.env")],
                ["Change a setting", "Edit " + C("/etc/komute/komute.env") + ". Then restart the service."]],
               [0.3, 0.7]),
          Spacer(1, 4),
          grid([["Path", "Contents"],
                [C("/opt/komute/app"), "The code, start.sh, setup.sh and the model files"],
                [C("/opt/komute/venv"), "Python 3.12 and the packages"],
                [C("/etc/komute/komute.env"), "Settings and secrets (owner root, group komute, mode 0640)"],
                [C("/var/lib/komute/storage"), "Encrypted session records and images (only the komute user can read it)"],
                [C("/etc/systemd/system/komute.service"), "The service definition"]],
               [0.42, 0.58])]

    s += [P("6. Settings in /etc/komute/komute.env", h2),
          grid([["Setting", "Value after setup", "Meaning"],
                [C("KOMUTE_ENV"), "dev", "dev skips liveness. production needs a liveness model and an API key "
                                         "of 24 or more characters. The node does not start in production without them."],
                [C("KOMUTE_DEV_SKIP_LIVENESS"), "true", "No liveness model exists yet. Responses contain LIVENESS_SKIPPED_DEV."],
                [C("KOMUTE_API_KEY"), "random, 43 characters", "Callers must send it in the X-API-Key header"],
                [C("KOMUTE_STORAGE_KEY_B64"), "random, 32 bytes", "Encrypts the stored images and records (AES-256-GCM)"],
                [C("KOMUTE_MANIFEST_KEY_B64"), "random, 32 bytes", "Signs each session record (HMAC-SHA256)"],
                [C("KOMUTE_FACE_MATCH_THRESHOLD"), "0.35", "Face match pass mark (cosine similarity)"],
                [C("KOMUTE_HOST"), "127.0.0.1", "Only programs on this server can call the node"],
                [C("KOMUTE_PORT"), "8000", "TCP port"]],
               [0.33, 0.2, 0.47]),
          Spacer(1, 3),
          P(f"Other settings and their defaults are in {C('/opt/komute/app/.env.example')}. "
            "Add a line to komute.env to change one. Then restart the service.", base),
          Spacer(1, 3),
          note("WARNING", "Back up /etc/komute/komute.env to a safe place. Without the storage key, nobody can read "
                          "the stored data. Set the final keys before the first real verification. "
                          "Do not change the storage key after the node has stored data."),
          Spacer(1, 3),
          note("CAUTION", "The keys come from the setup script. For production, put keys from your secret manager "
                          "into komute.env before the first real verification.")]

    s += [P("7. Update to a new version", h2)]
    s += steps(["Make and copy a new package (section 3).",
                "Extract it to a new folder on the server (section 4, steps 1 and 2).",
                (f"Run the setup script again: {C('sudo bash ./setup.sh')}. It replaces the code. "
                 "It keeps the settings, the keys, the models and the stored data. It restarts the service.", [])])

    # ---------------------------------------------------------------- integration
    s += [PageBreak(), P("8. Connect the main application", h2),
          P("The main application calls the node over HTTP. The node has three endpoints.", base), Spacer(1, 3),
          grid([["Endpoint", "Use"],
                [C("POST /v1/verify"), "Send one verification. The answer is the decision."],
                [C("GET /v1/sessions/{request_id}"), "Read the stored result of a finished request again"],
                [C("GET /health"), "Readiness check (no API key necessary). 200 = ready, 503 = not ready."]],
               [0.38, 0.62]),
          P("8.1 Network connection", h3),
          grid([["Main application location", "What to do"],
                ["Same server as the node (recommended)",
                 f"Use the base URL {C('http://127.0.0.1:8000')}. Do not open port 8000 in the firewall."],
                ["A different server",
                 f"Set {C('KOMUTE_HOST')} to the private network IP of the node. Let only the main application "
                 "server connect to port 8000 (for example: " + C("ufw allow from MAIN_APP_IP to any port 8000 proto tcp")
                 + "). Put a TLS reverse proxy in front of the node, because the policy needs TLS 1.3 in transit. "
                   "The scripts do not set up TLS."]],
               [0.3, 0.7]),
          Spacer(1, 3),
          note("WARNING", "Do not open port 8000 to the internet."),
          P("8.2 Configuration in the main application", h3),
          grid([["Setting", "Value"],
                ["Base URL", f"{C('http://127.0.0.1:8000')} (same server). Keep it in configuration, not in code."],
                ["API key", "The value of KOMUTE_API_KEY. Send it in the header X-API-Key."],
                ["Timeouts", "Connect: 3 seconds. Read: 30 seconds."],
                ["Busy retries", "3. Wait the Retry-After seconds (default 5) between tries."]],
               [0.25, 0.75])]

    s += [P("8.3 Inputs (the request)", h3),
          P(f"Send {C('POST /v1/verify')} as {C('multipart/form-data')}. Send the photos as binary file parts. "
            "Do not send base64 text.", base), Spacer(1, 3),
          grid([["Field", "Type", "Rule"],
                [C("request_id"), "text", "Unique for each attempt. Letters, digits, - and _. Maximum 64 characters."],
                [C("driver_id"), "text", "Same rules as request_id"],
                [C("consent_at"), "text", "Time of the driver's consent. ISO-8601 with a time zone, for example "
                                          + C("2026-11-11T12:00:00-05:00")],
                [C("consent_policy_version"), "text", "Version of the consent text, for example v1"],
                [C("claimed.full_name"), "text", "Full legal name"],
                [C("claimed.date_of_birth"), "text", "YYYY-MM-DD"],
                [C("claimed.licence_number"), "text", "Hyphens and spaces are permitted"],
                [C("claimed.issuing_province"), "text", "ON QC BC AB MB SK NB NS NL PE YT NT NU (NF is read as NL)"],
                [C("claimed.licence_class"), "text", "As printed on the licence, for example G or 5"],
                [C("claimed.issue_date") + "<br/>" + C("claimed.expiry_date"), "text", "YYYY-MM-DD"],
                [C("claimed.plate_number"), "text", "Spaces and hyphens are permitted"],
                [C("claimed.vehicle_colour"), "text", "For example Black. Advisory only."],
                [C("claimed.vehicle_make") + "<br/>" + C("claimed.vehicle_model") + "<br/>" + C("claimed.vehicle_year"),
                 "text", "Optional. The node stores them. It does not check them."],
                [C("selfie") + ", " + C("licence_front") + "<br/>" + C("licence_back") + ", " + C("vehicle"), "file",
                 "JPEG, PNG or WebP. Maximum 10 MB each. Send the original photo: no crop, no filter. "
                 "1200 pixels or more on the long side is recommended."]],
               [0.33, 0.08, 0.59])]

    s += [KeepTogether([P("8.4 Example (Python, requests)", h3),
          code('import requests\n'
               '\n'
               'fields = {\n'
               '    "request_id": "req-1001", "driver_id": "drv-42",\n'
               '    "consent_at": "2026-11-11T12:00:00-05:00", "consent_policy_version": "v1",\n'
               '    "claimed.full_name": "Alex Tester", "claimed.date_of_birth": "1990-03-15",\n'
               '    "claimed.licence_number": "T1234-56789-01234", "claimed.issuing_province": "ON",\n'
               '    "claimed.licence_class": "G", "claimed.issue_date": "2023-09-14",\n'
               '    "claimed.expiry_date": "2030-12-31",\n'
               '    "claimed.plate_number": "ABC 123", "claimed.vehicle_colour": "Grey",\n'
               '}\n'
               'with open("selfie.jpg", "rb") as s, open("front.jpg", "rb") as f, \\\n'
               '     open("back.jpg", "rb") as b, open("car.jpg", "rb") as v:\n'
               '    r = requests.post(f"{BASE_URL}/v1/verify", headers={"X-API-Key": API_KEY}, data=fields,\n'
               '                      files={"selfie": s, "licence_front": f, "licence_back": b, "vehicle": v},\n'
               '                      timeout=(3, 30))\n'
               'if r.status_code == 200:\n'
               '    result = r.json()          # result["decision"] is "APPROVED" or "REJECTED"')]),
          Spacer(1, 3),
          note("NOTE", "The node does one verification at a time. The main application already runs Django Q "
                       "(qcluster). A queue task with one worker is a good way to call the node.")]

    s += [P("8.5 What the node does with the inputs", h3),
          grid([["Stage", "Checks"],
                ["1. Licence barcode",
                 "Reads the PDF417 barcode on the licence back (AAMVA format). The barcode is the reference data. "
                 "Checks: province = barcode issuer; name (fuzzy match, 90% pass); date of birth, licence number, class, "
                 "issue date and expiry date (exact match); licence not expired; issue date not after today; age 18 or more. "
                 "If stage 1 fails, stages 2 and 3 do not run."],
                ["2. Face",
                 "Finds the face in the selfie and in the licence front photo (YuNet). Liveness check on the selfie "
                 "(skipped until a liveness model is installed). Compares the two faces (w600k_r50). "
                 "Pass: similarity 0.35 or more."],
                ["3. Vehicle",
                 "Finds the licence plate (YOLOv9-tiny) and reads it (A to Z, 0 to 9). Pass: the plate is equal to "
                 "the claimed plate, or one character differs by O/0, I/1 or B/8. The body colour is advisory only."],
                ["Record",
                 "Stores the photos encrypted (AES-256-GCM) and writes a signed record (HMAC-SHA256) for each session. "
                 "Deletes the images of rejected sessions within 72 hours. Deletes the original photos of approved "
                 "sessions after 30 days."]],
               [0.18, 0.82])]

    s += [P("8.6 Outputs (the response)", h3),
          P("HTTP 200 means that the node made a decision. Read the field decision.", base), Spacer(1, 3),
          code('{\n'
               '  "request_id": "req-1001",\n'
               '  "decision": "REJECTED",\n'
               '  "reasons": [\n'
               '    { "code": "BARCODE_UNREADABLE", "retryable": true,\n'
               '      "message": "The barcode on the back of your licence is not clearly visible. ..." }\n'
               '  ],\n'
               '  "stages": { "barcode": { "ran": true, ... }, "face": { "ran": false }, "vehicle": { "ran": false } },\n'
               '  "advisories": [],\n'
               '  "timings_ms": { "total": 2210.4, "barcode": 2100.0, "face": 0.0, "vehicle": 0.0 },\n'
               '  "rss_mb": 318.5,\n'
               '  "audit": { "manifest_id": "aa11...", "key_id": "m1" }\n'
               '}'),
          Spacer(1, 3),
          grid([["Field", "Meaning"],
                [C("decision"), "APPROVED or REJECTED. There is no other result and no manual review."],
                [C("reasons[]"), "Empty when approved. One entry for each failed check."],
                [C("reasons[].code"), "Stable code (table 8.8). Use the code in your logic, not the message."],
                [C("reasons[].retryable"), "true: a better photo can change the result. false: a new photo cannot."],
                [C("reasons[].message"), "A sentence that you can show to the driver"],
                [C("stages"), "Details of each stage. ran = false: an earlier stage failed, so this stage did not run."],
                [C("advisories[]"), "Information only. They do not stop an approval: COLOUR_MISMATCH, COLOUR_UNVERIFIED, "
                                    "LICENCE_FORMAT_UNUSUAL, LIVENESS_SKIPPED_DEV."],
                [C("audit.manifest_id"), "Store it with your record. It identifies the signed record of this decision."]],
               [0.25, 0.75])]

    s += [P("8.7 Errors (no decision)", h3),
          P(f"All errors have this shape: {C('{\"error\": {\"code\": \"...\", \"message\": \"...\"}}')}", base),
          Spacer(1, 3),
          grid([["HTTP", "error.code", "Action"],
                ["422", "MISSING_FIELD, INVALID_FIELD, CONSENT_REQUIRED, INVALID_IMAGE, IMAGE_TOO_LARGE",
                 "Correct the request. Send it again with a new request_id."],
                ["401", "UNAUTHORIZED", "Send the correct X-API-Key header"],
                ["409", "REQUEST_ID_CONFLICT", "This request_id was used with different data. Use a new request_id."],
                ["503", "BUSY", "A verification is in progress. Wait Retry-After seconds. Then try again."],
                ["500", "ENGINE_ERROR", "The node failed. This is not a rejection. Try again later. "
                                        "Tell the operator if it occurs again."],
                ["none", "network error or timeout", f"This is not a rejection. Try again. Do a check of {C('GET /health')}."]],
               [0.08, 0.42, 0.5]),
          Spacer(1, 3),
          note("WARNING", "Do not record a non-200 answer as REJECTED. Only HTTP 200 with decision REJECTED is a rejection.")]

    s += [P("8.8 Reason codes", h3),
          grid([["Code", "Retryable", "Tell the driver"],
                ["BARCODE_UNREADABLE, BARCODE_NOT_AAMVA", "yes", "Take the photo of the licence back again"],
                ["FACE_NOT_FOUND_SELFIE, LIVENESS_FAILED, IMAGE_QUALITY", "yes", "Take the selfie again"],
                ["FACE_NOT_FOUND_LICENCE", "yes", "Take the photo of the licence front again"],
                ["PLATE_NOT_FOUND, PLATE_UNREADABLE", "yes", "Take the vehicle photo again"],
                ["JURISDICTION_MISMATCH, NAME_MISMATCH, DOB_MISMATCH, LICENCE_NUMBER_MISMATCH, CLASS_MISMATCH, "
                 "ISSUE_DATE_MISMATCH, EXPIRY_DATE_MISMATCH", "no", "The typed data does not agree with the licence"],
                ["LICENCE_EXPIRED, UNDERAGE, LICENCE_NOT_YET_VALID", "no", "The driver is not eligible"],
                ["FACE_MISMATCH", "no", "The selfie does not agree with the licence photo"],
                ["PLATE_MISMATCH", "no", "The plate in the photo does not agree with the typed plate"]],
               [0.55, 0.1, 0.35])]

    s += [P("8.9 Rules for the main application", h3)]
    s += steps([
        "Make a new request_id for each new attempt, for example after a new photo.",
        "After a timeout, send the same request_id with the same data. The node gives the stored result. It does not run again.",
        "Do not send the same request_id with different data. The node gives 409.",
        "Send one verification at a time to one node. On 503, wait Retry-After seconds and try again.",
        "Do a check of GET /health before you send traffic.",
        "Store decision, request_id, reasons[].code and audit.manifest_id. These contain no images.",
        "Do not log request bodies. They contain personal data.",
        "Get the driver's consent in the main application before you call the node.",
    ])
    s += [Spacer(1, 3),
          P("A verification takes approximately 1.5 to 2 seconds on the development laptop. A photo with no readable "
            "barcode can take up to approximately 10 seconds. On a 1 vCPU server, expect longer times.", base)]

    s += [P("9. Limits", h2)]
    s += bullets([
        "No liveness model is installed. The node cannot detect a printed photo or a screen in front of the camera.",
        "The face pass mark 0.35 is set from a public test set (AgeDB-30: 0 of 3,000 wrong people accepted). "
        "Confirm it on real licence and selfie pairs before production.",
        "Memory use on Linux is not yet measured.",
        "TLS is not set up by the scripts. Use the same server, or add a TLS reverse proxy.",
    ])
    return s


if __name__ == "__main__":
    build("komute-local-windows-setup.pdf", "Windows laptop setup", windows())
    build("komute-local-linux-deploy-and-integration.pdf", "Linux deployment and integration", linux())
