# Error Analysis

## Always Cheap

- Errors: 56 / 240
- By category: {'text': 25, 'tool': 12, 'vision': 19}
- Top datasets: {'gsm8k': 22, 'mmmu': 7, 'ocr-vqa': 6, 'bfcl-simple_python': 6, 'bfcl-multiple': 6}

## Always Strong

- Errors: 53 / 240
- By category: {'text': 19, 'tool': 16, 'vision': 18}
- Top datasets: {'gsm8k': 18, 'bfcl-simple_python': 9, 'bfcl-multiple': 7, 'mmmu': 6, 'ocr-vqa': 5}

## Task-Aware Router

- Errors: 48 / 240
- By category: {'text': 19, 'tool': 14, 'vision': 15}
- Top datasets: {'gsm8k': 18, 'bfcl-simple_python': 7, 'bfcl-multiple': 7, 'ocr-vqa': 5, 'mmmu': 5}

## Reflection Router

- Errors: 56 / 240
- By category: {'text': 25, 'tool': 12, 'vision': 19}
- Top datasets: {'gsm8k': 22, 'mmmu': 7, 'ocr-vqa': 6, 'bfcl-simple_python': 6, 'bfcl-multiple': 6}

## Reflection Diagnostics

- Both models correct: 170
- Strong fixes a cheap-model error: 17
- Strong regresses a correct cheap-model answer: 14
- Both models wrong: 39
- False accepts (cheap answer wrong but accepted): 51
- Unnecessary escalations (cheap answer correct but escalated): 0
- Beneficial escalations (cheap wrong, final correct): 0
- Harmful escalations (cheap correct, final wrong): 0
- Review actions: {'correct': 3, 'keep': 2}

### Representative False Accepts

- `gsm8k-0026` (gsm8k): calibrated_probability=0.5054178091774851, reason=`accepted`
- `ocr-vqa-0012` (ocr-vqa): calibrated_probability=0.8214810366870134, reason=`accepted`
- `ocr-vqa-0001` (ocr-vqa): calibrated_probability=0.8420147335245876, reason=`accepted`
- `bfcl-simple_python-0034` (bfcl-simple_python): calibrated_probability=0.9544113201305149, reason=`accepted`
- `mmmu-0016` (mmmu): calibrated_probability=0.9506759693210901, reason=`accepted`
- `gsm8k-0023` (gsm8k): calibrated_probability=0.7251929037884027, reason=`accepted`
- `bfcl-multiple-0045` (bfcl-multiple): calibrated_probability=0.9605453606025787, reason=`accepted`
- `gsm8k-0036` (gsm8k): calibrated_probability=0.6735190096637172, reason=`accepted`
