# Error Analysis

## Always Cheap

- Errors: 48 / 240
- By category: {'text': 24, 'tool': 13, 'vision': 11}
- Top datasets: {'gsm8k': 22, 'bfcl-simple_python': 7, 'scienceqa-img': 7, 'bfcl-multiple': 6, 'ocr-vqa': 3}

## Always Strong

- Errors: 44 / 240
- By category: {'text': 19, 'tool': 16, 'vision': 9}
- Top datasets: {'gsm8k': 18, 'bfcl-simple_python': 9, 'bfcl-multiple': 7, 'scienceqa-img': 5, 'ocr-vqa': 3}

## Task-Aware Router

- Errors: 45 / 240
- By category: {'text': 20, 'tool': 14, 'vision': 11}
- Top datasets: {'gsm8k': 18, 'bfcl-simple_python': 7, 'bfcl-multiple': 7, 'scienceqa-img': 7, 'ocr-vqa': 3}

## Reflection Router

- Errors: 51 / 240
- By category: {'text': 24, 'tool': 16, 'vision': 11}
- Top datasets: {'gsm8k': 22, 'bfcl-simple_python': 9, 'bfcl-multiple': 7, 'scienceqa-img': 7, 'ocr-vqa': 3}

## Reflection Diagnostics

- Both models correct: 183
- Strong fixes a cheap-model error: 13
- Strong regresses a correct cheap-model answer: 9
- Both models wrong: 35
- False accepts (cheap answer wrong but accepted): 35
- Unnecessary escalations (cheap answer correct but escalated): 67

### Representative False Accepts

- `gsm8k-0035` (gsm8k): confidence=1.0, reason=`accepted`
- `gsm8k-0026` (gsm8k): confidence=1.0, reason=`accepted`
- `gsm8k-0023` (gsm8k): confidence=1.0, reason=`accepted`
- `gsm8k-0027` (gsm8k): confidence=0.95, reason=`accepted`
- `ocr-vqa-0003` (ocr-vqa): confidence=0.9, reason=`accepted`
- `gsm8k-0036` (gsm8k): confidence=1.0, reason=`accepted`
- `gsm8k-0000` (gsm8k): confidence=1.0, reason=`accepted`
- `scienceqa-0076` (scienceqa-img): confidence=0.95, reason=`accepted`
