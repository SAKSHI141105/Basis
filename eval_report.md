# Evaluation Report

Golden set size: **198**

## Intent classification

| System | Accuracy | Macro-F1 |
|---|---|---|
| trivial | 0.051 | 0.011 |
| simple | 0.394 | 0.416 |
| main | 0.803 | 0.807 |

## Escalation decision

| System | Precision | Recall | F1 | Cost-weighted |
|---|---|---|---|---|
| trivial | 0.414 | 1.000 | 0.586 | 0.805 |
| simple | 0.750 | 0.037 | 0.070 | 0.599 |
| main | 0.433 | 0.707 | 0.537 | 0.751 |

## LLM judge (reply quality, 1-5)

- mean_groundedness: 4.73
- mean_correctness: 4.74
- mean_tone: 4.88
- mean_actionability: 4.61
- mean_overall: 4.74

## Human-agreement study

Not run — no human judge scores found at `artifacts\human_judge_scores.json`. TRD 7.3 marks this mandatory before trusting the judge for headline numbers.

## What's misleading about my headline number

What would be misleading here is assuming this golden set reflects real traffic volume: it is deliberately stratified roughly evenly across intents (pipeline/golden_set.py), NOT the real ~88% out_of_scope-skewed traffic distribution (taxonomy.yaml) -- so the trivial baseline's accuracy here (0.051) is low, not falsely high. On real, naturally-imbalanced traffic the same trivial baseline would score far higher on accuracy while being equally useless. Read accuracy numbers from this report as 'performance on a balanced sample,' not 'performance on real traffic volume.' See REPORT.md section 8.

## Top failure examples

- **1236865_1236864**: "iOS 11 is buggy. Please fix it. 😩"
  - intent: true=`device_troubleshooting` pred=`out_of_scope`, escalation: true=`auto_handle` pred=`escalate`
- **2000150_2000149**: "your new IOS update is complete fucking dog shit. Ever since i downloaded this bs my phone hasnt work properly. I want the old 1 bak"
  - intent: true=`device_troubleshooting` pred=`software_update_bug`, escalation: true=`escalate` pred=`auto_handle`
- **2406550_2406548**: "My iPhone 6 went dead while I was transferring it to a new iCloud account? Any obvious reasons as to why this might have happened? Never seen an iPhone going dead and it was supposed to be 2 years old (bought it 2 years ago). Thanks"
  - intent: true=`software_update_bug` pred=`device_troubleshooting`, escalation: true=`auto_handle` pred=`escalate`
- **2422587_2422584**: "Ever since the update my reminders display on my lock screen all the time. Even before and after they’re due. And even when marked completed. What’s the deal #AppleSupport #iPhoneUpdate"
  - intent: true=`general_complaint` pred=`software_update_bug`, escalation: true=`escalate` pred=`escalate`
- **622711_622712**: "11.1.2 (15B202). USA"
  - intent: true=`feature_how_to` pred=`out_of_scope`, escalation: true=`escalate` pred=`escalate`
