# V3 Error Analysis

## Always Cheap

- Errors: 37 / 150
- By category: {'text': 13, 'tool': 10, 'vision': 14}
- Top datasets: {'gsm8k': 10, 'ocr-vqa': 7, 'bfcl-simple_python': 5, 'bfcl-multiple': 5, 'scienceqa-img': 3}

## Always Strong

- Errors: 31 / 150
- By category: {'text': 10, 'tool': 9, 'vision': 12}
- Top datasets: {'gsm8k': 6, 'ocr-vqa': 6, 'bfcl-simple_python': 5, 'mmmu': 4, 'bfcl-multiple': 4}

## Handcrafted Task-Aware

- Errors: 32 / 150
- By category: {'text': 10, 'tool': 9, 'vision': 13}
- Top datasets: {'gsm8k': 6, 'ocr-vqa': 6, 'bfcl-simple_python': 5, 'mmmu': 5, 'bfcl-multiple': 4}

## Learned Cost-Aware

- Errors: 32 / 150
- By category: {'text': 11, 'tool': 8, 'vision': 13}
- Top datasets: {'gsm8k': 7, 'ocr-vqa': 6, 'commonsenseqa': 4, 'mmmu': 4, 'bfcl-simple_python': 4}

## Calibrated Reflection

- Errors: 39 / 150
- By category: {'text': 12, 'tool': 10, 'vision': 17}
- Top datasets: {'gsm8k': 8, 'ocr-vqa': 7, 'bfcl-simple_python': 5, 'mmmu': 5, 'bfcl-multiple': 5}

## Model Pair

- Both correct: 106
- Strong only correct: 13
- Cheap only correct: 7
- Both wrong: 24
