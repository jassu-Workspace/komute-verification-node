# Canadian Multi-Province Brutal Driving Licence OCR Audit Report

**Date:** 2026-10-12T16:14:48.418805  
**Hardware Platform:** 12th Gen Intel(R) Core(TM) i7-1255U (8 Worker Threads) + Intel(R) Iris(R) Xe Graphics  
**Total Test Cases:** 1326 across 13 issuing jurisdictions  
**Overall Pass Rate:** 71.04% (Benchmark: 80.0%)  
**National Verdict:** **FAILED**  

---

## Master Scorecard by Jurisdiction

| Region | Jurisdiction | Cases | Passed | Failed | Pass Rate | Field Accuracy | Regional Threshold | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AB` | Alberta | 102 | 89 | 13 | 87.25% | 93.53% | 92.0% | **FAILED** |
| `BC` | British Columbia | 102 | 82 | 20 | 80.39% | 90.00% | 92.0% | **FAILED** |
| `MB` | Manitoba | 102 | 16 | 86 | 15.69% | 78.63% | 92.0% | **FAILED** |
| `NB` | New Brunswick | 102 | 79 | 23 | 77.45% | 88.63% | 92.0% | **FAILED** |
| `NL` | Newfoundland & Labrador | 102 | 84 | 18 | 82.35% | 91.96% | 92.0% | **FAILED** |
| `NS` | Nova Scotia | 102 | 83 | 19 | 81.37% | 91.18% | 92.0% | **FAILED** |
| `NT` | Northwest Territories | 102 | 66 | 36 | 64.71% | 89.02% | 92.0% | **FAILED** |
| `NU` | Nunavut | 102 | 86 | 16 | 84.31% | 91.37% | 92.0% | **FAILED** |
| `ON` | Ontario | 102 | 15 | 87 | 14.71% | 76.27% | 92.0% | **FAILED** |
| `PE` | Prince Edward Island | 102 | 83 | 19 | 81.37% | 91.37% | 92.0% | **FAILED** |
| `QC` | Quebec | 102 | 84 | 18 | 82.35% | 91.18% | 92.0% | **FAILED** |
| `SK` | Saskatchewan | 102 | 88 | 14 | 86.27% | 92.35% | 92.0% | **FAILED** |
| `YT` | Yukon | 102 | 87 | 15 | 85.29% | 91.96% | 92.0% | **FAILED** |
| **TOTAL** | **All Jurisdictions** | **1326** | **942** | **384** | **71.04%** | - | **80.0%** | **FAILED** |

---

## Regional Breakdown

### Passed Jurisdictions (0)

### Failed Jurisdictions (13)
- **AB** (Alberta): **87.25%** (13 failures)
- **BC** (British Columbia): **80.39%** (20 failures)
- **MB** (Manitoba): **15.69%** (86 failures)
- **NB** (New Brunswick): **77.45%** (23 failures)
- **NL** (Newfoundland & Labrador): **82.35%** (18 failures)
- **NS** (Nova Scotia): **81.37%** (19 failures)
- **NT** (Northwest Territories): **64.71%** (36 failures)
- **NU** (Nunavut): **84.31%** (16 failures)
- **ON** (Ontario): **14.71%** (87 failures)
- **PE** (Prince Edward Island): **81.37%** (19 failures)
- **QC** (Quebec): **82.35%** (18 failures)
- **SK** (Saskatchewan): **86.27%** (14 failures)
- **YT** (Yukon): **85.29%** (15 failures)

---

## Sample Comparative Verification Proofs (Expected vs. Extracted)

| Region | Case | Category | Expected DL | Extracted DL | Expected Name | Extracted Name | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `AB` | `test_001.png` | Clean Base | `134711320` | `134711-320` | `SAM SAMPLE` | `Sam Sample` | PASS |
| `AB` | `test_002.png` | Clean Base | `987654321` | `987654-321` | `GORDON VANCE` | `Gordon Vance` | PASS |
| `AB` | `test_003.png` | Clean Base | `123456789` | `123456-789` | `TRAVIS SCOTT` | `Travis Scott` | PASS |
| `BC` | `test_001.png` | Clean Base | `98765432` | `98765432` | `JANE SMITH` | `Jane Smith` | PASS |
| `BC` | `test_002.png` | Clean Base | `12345678` | `12345678` | `ROBERT CHEN` | `Robert Chen` | PASS |
| `BC` | `test_003.png` | Clean Base | `87654321` | `87654321` | `LIAM WILSON` | `Liam Wilson` | PASS |
| `MB` | `test_001.png` | Clean Base | `PUBLIJQ123AB` | `PUBLI-JQ123-AB` | `JAMES BEAZLEY` | `James Beazley` | PASS |
| `MB` | `test_002.png` | Clean Base | `KUROS12345CD` | `KUROS-12345-CD` | `AKIRA KUROSAWA` | `Akira Kurosawa` | FAIL |
| `MB` | `test_003.png` | Clean Base | `COOPE67890EF` | `COOPE-67890-EF` | `BRADLEY COOPER` | `Bradley Cooper` | FAIL |
| `NB` | `test_001.png` | Clean Base | `1234567` | `1234567` | `PAUL ATKINSON` | `Paul Atkinson` | PASS |
| `NB` | `test_002.png` | Clean Base | `7654321` | `7654321` | `MARTIN LEBLANC` | `Martin LeBlanc` | PASS |
| `NB` | `test_003.png` | Clean Base | `2345678` | `2345678` | `CLAUDE CORMIER` | `Claude Cormier` | PASS |
| `NL` | `test_001.png` | Clean Base | `M780515123` | `M-780515-123` | `PETER MILLER` | `Peter Miller` | PASS |
| `NL` | `test_002.png` | Clean Base | `P990917999` | `P-990917-999` | `GORDON PIKE` | `Gordon Pike` | PASS |
| `NL` | `test_003.png` | Clean Base | `B850412456` | `B-850412-456` | `ARTHUR BROWN` | `Arthur Brown` | PASS |
| `NS` | `test_001.png` | Clean Base | `SANTA150378123` | `SANTA-150378-123` | `CARLOS SANTANA` | `Carlos Santana` | PASS |
| `NS` | `test_002.png` | Clean Base | `MAHI*020280456` | `MAHI*-020280-456` | `GILL MAHI` | `Gill Mahi` | PASS |
| `NS` | `test_003.png` | Clean Base | `YOUNG120485789` | `YOUNG-120485-789` | `ANGUS YOUNG` | `Angus Young` | PASS |
| `NT` | `test_001.png` | Clean Base | `123456789` | `123456789` | `LIAM VANCE` | `Liam Vance` | PASS |
| `NT` | `test_002.png` | Clean Base | `987654321` | `987654321` | `ETHAN FROST` | `Ethan Frost` | PASS |
| `NT` | `test_003.png` | Clean Base | `234567891` | `234567891` | `CHLOE WINTER` | `Chloe Winter` | PASS |
| `NU` | `test_001.png` | Clean Base | `123456` | `123456` | `NOAH IQALUIT` | `Noah Iqaluit` | PASS |
| `NU` | `test_002.png` | Clean Base | `654321` | `654321` | `MARY SIMON` | `Mary Simon` | PASS |
| `NU` | `test_003.png` | Clean Base | `234567` | `234567` | `PAUL QUASSA` | `Paul Quassa` | PASS |
| `ON` | `test_001.png` | Clean Base | `D61014070660905` | `D6101-40706-60905` | `JOHN DOE` | `John Doe` | PASS |
| `ON` | `test_002.png` | Clean Base | `S12345678901234` | `S1234-56789-01234` | `ALICE SMITH` | `Alice Smith` | FAIL |
| `ON` | `test_003.png` | Clean Base | `B54321098765432` | `B5432-10987-65432` | `MICHAEL BROWN` | `Michael Brown` | FAIL |
| `PE` | `test_001.png` | Clean Base | `123456` | `123456` | `ADA LOVELACE` | `Ada Lovelace` | PASS |
| `PE` | `test_002.png` | Clean Base | `654321` | `654321` | `LUCY MONTGOMERY` | `Lucy Montgomery` | PASS |
| `PE` | `test_003.png` | Clean Base | `234567` | `234567` | `ANNE SHIRLEY` | `Anne Shirley` | PASS |
| `QC` | `test_001.png` | Clean Base | `C123415029001` | `C1234-150290-01` | `MARIE CURIE` | `Marie Curie` | PASS |
| `QC` | `test_002.png` | Clean Base | `T567812048502` | `T5678-120485-02` | `JEAN TREMBLAY` | `Jean Tremblay` | PASS |
| `QC` | `test_003.png` | Clean Base | `G901225117803` | `G9012-251178-03` | `PIERRE GAGNON` | `Pierre Gagnon` | PASS |
| `SK` | `test_001.png` | Clean Base | `20000030` | `20000030` | `GRACE HOPPER` | `Grace Hopper` | PASS |
| `SK` | `test_002.png` | Clean Base | `87654321` | `87654321` | `SCOTT SAMPLE` | `Scott Sample` | PASS |
| `SK` | `test_003.png` | Clean Base | `12345678` | `12345678` | `BARRY BONDS` | `Barry Bonds` | PASS |
| `YT` | `test_001.png` | Clean Base | `654321` | `654321` | `DAVID DAWSON` | `David Dawson` | PASS |
| `YT` | `test_002.png` | Clean Base | `123456` | `123456` | `JACK LONDON` | `Jack London` | PASS |
| `YT` | `test_003.png` | Clean Base | `234567` | `234567` | `ROBERT SERVICE` | `Robert Service` | PASS |