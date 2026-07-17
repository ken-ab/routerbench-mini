# V5 Failure Analysis

- Learned missed beneficial Strong upgrades: 6
- Learned selected harmful Strong regressions: 28
- Reflection false accepts among review-fixable cases: 0
- Reflection harmful final answers after a correct Cheap answer: 13

## Example IDs

- Learned missed: ['v5:bfcl_multiple:BFCL_v4_live_multiple:live_multiple_652-161-20', 'v5:bfcl_simple:BFCL_v4_live_simple:live_simple_156-95-13', 'v5:gsm8k:test:372', 'v5:gsm8k:test:497', 'v5:gsm8k:test:553', 'v5:gsm8k:test:571']
- Learned harmful: ['v5:bbh:boolean_expressions:194', 'v5:bbh:date_understanding:121', 'v5:bbh:tracking_shuffled_objects_seven_objects:143', 'v5:bbh:tracking_shuffled_objects_seven_objects:155', 'v5:bbh:tracking_shuffled_objects_seven_objects:38', 'v5:bfcl_simple:BFCL_v4_live_simple:live_simple_144-95-1', 'v5:bfcl_simple:BFCL_v4_simple_java:simple_java_81', 'v5:chartqa:test:1967', 'v5:chartqa:test:299', 'v5:chartqa:test:353', 'v5:chartqa:test:865', 'v5:commonsenseqa:b5baf77d3855935c87f01f5fb2216667', 'v5:commonsenseqa:c68b4082a6872cf8198502651d0f3352', 'v5:gsm8k:test:1031', 'v5:gsm8k:test:1083', 'v5:gsm8k:test:1199', 'v5:gsm8k:test:13', 'v5:gsm8k:test:575', 'v5:gsm8k:test:592', 'v5:gsm8k:test:642']
- Reflection missed: []
- Reflection harmful: ['v5:bbh:boolean_expressions:194', 'v5:bbh:date_understanding:121', 'v5:bbh:tracking_shuffled_objects_seven_objects:38', 'v5:chartqa:test:1967', 'v5:chartqa:test:353', 'v5:chartqa:test:865', 'v5:commonsenseqa:b5baf77d3855935c87f01f5fb2216667', 'v5:commonsenseqa:c68b4082a6872cf8198502651d0f3352', 'v5:gsm8k:test:1083', 'v5:gsm8k:test:13', 'v5:gsm8k:test:729', 'v5:scienceqa:test:1335', 'v5:scienceqa:test:247']

The corresponding raw outputs and grader fields are in `test_model_outputs.jsonl` and `test_review_outputs.jsonl`.
