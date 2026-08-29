import re

def get_description(name):
    if "Easy Standard License" in name:
        return "Baseline test with a clean, unaugmented image to verify standard text extraction.", "None (Clean Image)"
    elif "Blur Level" in name:
        # Extract level to give context
        level = "".join(filter(str.isdigit, name.split("Level")[-1]))
        return "Applies Gaussian blur to simulate camera out-of-focus scenarios.", f"Gaussian Blur (Kernel size proportional to level {level})"
    elif "Gaussian Noise" in name:
        severity = "".join(filter(str.isdigit, name.split("Severity")[-1]))
        return "Injects static Gaussian noise to simulate low-light or poor sensor artifacts.", f"Gaussian Noise (Severity: {severity})"
    elif "Rotation" in name:
        match = re.search(r'Rotation (-?\d+) degrees', name)
        deg = match.group(1) if match else "X"
        return "Rotates the image to simulate crooked scanning or careless photography.", f"Rotation ({deg}°)"
    elif "Intense Glare" in name:
        return "Overlays a solid white geometric shape with alpha blending to simulate intense camera flash or sunlight reflection.", "Artificial Glare Overlay (Opacity: 40%)"
    elif "Low Contrast" in name:
        return "Compresses the pixel intensity range to simulate washed-out lighting or poor exposure.", "Scale Abs (Alpha=0.5, Beta=128)"
    elif "Perspective Skew" in name:
        return "Applies a 3D perspective warp to simulate an ID card photographed at a sharp angle.", "Perspective Transform Matrix"
    elif "BRUTAL Combo" in name:
        return "Ultimate stress test combining multiple severe augmentations simultaneously to test failure limits.", "Blur (Lvl 3) + Noise (Sev 20) + Rotate (-5°) + 3D Skew"
    return "Standard test case evaluation.", "N/A"

def generate_md_report():
    input_file = "test_results.txt"
    output_file = "OCR_Test_Report.md"
    
    try:
        with open(input_file, 'r', encoding='utf-16le') as f:
            lines = f.readlines()
    except UnicodeError:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    md_content = [
        "# 📄 OCR Model Resilience Test Report\n",
        "**Total Test Cases:** 102  \n",
        "**Objective:** Evaluate the maximum potential and limits of the DL Verification Node OCR Engine under varying conditions (Blur, Noise, Rotation, Glare, Perspective Skew, and Combinations).\n",
        "---\n\n",
        "## 🏆 Overall Results\n"
    ]

    summary_lines = []
    test_cases = []
    
    for line in lines:
        line = line.strip()
        # Remove ANSI color codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        line = ansi_escape.sub('', line)
        
        if "TEST SUITE COMPLETED" in line or "OVERALL OCR RESILIENCE SCORE" in line:
            summary_lines.append(line.strip())
            
        elif line.startswith("[") and "/102]" in line:
            # Parse test case line
            parts = line.split(" | ")
            if len(parts) >= 4:
                prefix = parts[0].strip()
                status = parts[1].replace("Status: ", "").strip()
                conf = parts[2].replace("Conf: ", "").strip()
                details = parts[3].replace("Details: ", "").strip()
                
                # Get rich descriptions
                desc, params = get_description(prefix)
                
                # Format status
                if status == "PASSED":
                    status_md = "🟢 **PASSED**"
                else:
                    status_md = "🔴 **FAILED**"
                
                tc_md = f"### {prefix}\n\n"
                tc_md += f"**Description:** {desc}  \n"
                tc_md += f"**Applied Parameters:** `{params}`  \n"
                tc_md += f"**Status:** {status_md}  \n"
                tc_md += f"**Confidence:** {conf}  \n"
                tc_md += f"**Field Breakdown:** {details}  \n\n"
                
                # Extract index e.g., '[001' -> '001'
                idx_str = prefix.split('/')[0].replace('[', '')
                tc_md += f"**Visual Preview:**\n"
                tc_md += f"![Test Image Preview](test_images/test_{idx_str}.jpg)\n\n"
                
                tc_md += "---\n"
                test_cases.append(tc_md)

    for sl in summary_lines:
        md_content.append(f"> **{sl}**\n")
        
    md_content.append("\n## 🔍 Detailed Test Case Breakdown\n\n")
    md_content.extend(test_cases)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_content))
    
    print(f"Successfully generated {output_file}")

if __name__ == "__main__":
    generate_md_report()
