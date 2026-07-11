# V4 Error Analysis

## Always Cheap

- Errors: 32 / 150
- By category: {'text': 13, 'tool': 7, 'vision': 12}
- Top datasets: {'gsm8k': 10, 'chartqa': 4, 'bfcl-multiple': 4, 'mmmu': 3, 'commonsenseqa': 3}

## Always Strong

- Errors: 25 / 150
- By category: {'text': 7, 'tool': 7, 'vision': 11}
- Top datasets: {'gsm8k': 6, 'bfcl-multiple': 4, 'mmmu': 3, 'chartqa': 3, 'bfcl-simple_python': 3}

## Handcrafted Task-Aware

- Errors: 26 / 150
- By category: {'text': 9, 'tool': 7, 'vision': 10}
- Top datasets: {'gsm8k': 6, 'bfcl-multiple': 4, 'mmmu': 3, 'commonsenseqa': 3, 'bfcl-simple_python': 3}

## Learned Cost-Aware

- Errors: 30 / 150
- By category: {'text': 11, 'tool': 7, 'vision': 12}
- Top datasets: {'gsm8k': 9, 'chartqa': 4, 'bfcl-multiple': 4, 'mmmu': 3, 'bfcl-simple_python': 3}

## Calibrated Reflection

- Errors: 30 / 150
- By category: {'text': 9, 'tool': 7, 'vision': 14}
- Top datasets: {'chartqa': 5, 'gsm8k': 5, 'mmmu': 4, 'bfcl-multiple': 4, 'commonsenseqa': 3}

## Model Pair

- Both correct: 113
- Strong only correct: 12
- Cheap only correct: 5
- Both wrong: 20
