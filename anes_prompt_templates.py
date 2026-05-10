"""Prompt templates extracted from anes_pipeline.py."""

from typing import Dict

SYSTEM_PROMPT = """
You are a senior anesthesiologist.
Given de-identified structured intraoperative data, answer in Chinese.
If you need internal reasoning, keep it short inside <think>...</think> (max 3 lines).
After </think>, output EXACTLY ONE QA pair in strict format:
Q: ...
A: 【临床推理】：...
【决策干预（Miller）】：...
【决策干预（VitalDB）】：...
Q must be objective only (background + recent physiologic values/trends + intervention question), with no subjective clinical interpretation hints.
In 【决策干预（Miller）】, use three-part structure: 诊断依据：...; 具体干预：...; 原文摘录："...[M10#...]".
In 【决策干预（VitalDB）】, output an executable action order (drug + direction + magnitude/target + reassessment time + escalation/stop condition) with normalized units.
Do not output any bullets, headings, checklists, drafting notes, or instruction echoes.
""".strip()

# Golden-action keyword constants moved to anes_medication_constants.py

FEWSHOT_BY_TYPE: Dict[str, str] = {
    "continuous_infusion": (
        "### Example (continuous_infusion)\n"
        "<think>患者胸外科术中，近5分钟 MAP 下行而 BIS 上升，提示麻醉深度与血流动力学存在冲突。"
        "先稳灌注，再小步调整镇静药速率。</think>\n"
        "Q: 67岁男性，ASA III，胸外科维持期，近5分钟 MAP 72→58 mmHg、HR 86→102 bpm、SpO2 98→97%、BIS 52→66，结合手术背景，此时最合理的干预措施是什么？\n"
        "A: 【临床推理】：当前关键矛盾是循环稳定性与麻醉深度的平衡。若在低灌注状态下盲目加深镇静，可能进一步加重低血压并影响器官灌注。\n"
        "【决策干预（Miller）】：诊断依据：MAP持续低于65 mmHg且BIS上行; 具体干预：先滴定升压药0.1-0.3 mL/h并2 min复评，MAP≥65 mmHg后再小步调整镇静; 原文摘录:\"treat hypotension before deepening anesthesia\" [M10#1 | 术中相关章节: Hemodynamic management | p.1493]。\n"
        "【决策干预（VitalDB）】：立即按logged_action同类升压药将泵速上调0.1-0.3 mL/h，目标MAP≥65 mmHg；2 min复评MAP/HR，若MAP仍<65 mmHg再上调同幅度，若HR>110 bpm或MAP>85 mmHg则回调0.1 mL/h。\n"
        "### End Example\n"
    ),
    "bolus_like_event": (
        "### Example (bolus_like_event)\n"
        "<think>患者短时刺激期体征上冲，单次追加药物应以短效、可回退为原则。需避免过度镇静后低血压。</think>\n"
        "Q: 54岁女性，腹部手术刺激期，近3分钟 MAP 78→84 mmHg、HR 78→108 bpm、SpO2 99→99%、BIS 47→64，结合手术背景，此时最合理的干预措施是什么？\n"
        "A: 【临床推理】：短时、可逆的生理波动更适合短效追加干预；持续上调可能带来过量风险。需要结合血压、心率与麻醉深度的同步变化判断。\n"
        "【决策干预（Miller）】：诊断依据：BIS和HR同步上冲且MAP未低于65 mmHg; 具体干预：同类短效药单次追加0.5-1.0 mL，1-2 min复评后决定是否再追加0.5 mL; 原文摘录:\"short-acting incremental dosing with rapid reassessment\" [M10#2 | 术中相关章节: Analgesic titration | p.1521]。\n"
        "【决策干预（VitalDB）】：先按logged_action同类药物单次追加0.5-1.0 mL，再观察1-2 min；若BIS仍>60或HR>100 bpm则再追加0.5 mL，若MAP降至<65 mmHg则停止追加并改为维持泵速。\n"
        "### End Example\n"
    ),
    "arrhythmia_event": (
        "### Example (arrhythmia_event)\n"
        "<think>出现心律事件时，先判断血流动力学稳定性，再决定是否立即药理/电复律路径。麻醉深度与氧合通气也需并行评估。</think>\n"
        "Q: 69岁男性，泌尿外科术中突发心律失常标注，当前 MAP 62 mmHg、HR 42 bpm、SpO2 95%、BIS 45，且近2分钟MAP与HR均下降，结合手术背景，此时最合理的干预措施是什么？\n"
        "A: 【临床推理】：处理顺序应先看灌注与血压稳定性，再区分可观察与需立即干预的节律。同时排查缺氧、二氧化碳潴留、电解质异常及麻醉深度不匹配。\n"
        "【决策干预（Miller）】：诊断依据：心律事件伴MAP<65 mmHg和HR<50 bpm; 具体干预：先执行不稳定节律路径并给予同类急救药物追加0.5 mL，30-60 s复评后再决定升级; 原文摘录:\"hemodynamic instability determines urgency of treatment\" [M10#1 | 术中相关章节: Perioperative arrhythmia | p.1608]。\n"
        "【决策干预（VitalDB）】：若持续MAP<65 mmHg且HR<50 bpm，先给予同类急救药物追加0.5 mL并准备升级流程，30-60 s复评；若MAP回升≥65 mmHg则转入保守滴定并每2 min复评。\n"
        "### End Example\n"
    ),
    "unlabeled_context_snapshot": (
        "### Example (unlabeled_context_snapshot)\n"
        "<think>无明确事件标签时，依据趋势而非单点，优先识别威胁灌注与氧合的指标。在信息不全时给出保守且可复评的决策。</think>\n"
        "Q: 61岁女性，骨科维持期无明确事件标签，近5分钟 MAP 70→63 mmHg、HR 76→82 bpm、SpO2 98→96%、BIS 43→41，结合手术背景，此时最合理的干预措施是什么？\n"
        "A: 【临床推理】：应以 MAP/SpO2/HR 的连续趋势为主线，避免仅凭单一瞬时异常下结论。信息缺失时优先采取可逆、可滴定的策略。\n"
        "【决策干预（Miller）】：诊断依据：MAP持续下降并接近65 mmHg阈值; 具体干预：先小步调整同类循环支持0.1-0.2 mL/h并2 min复评，必要时再加0.1 mL/h; 原文摘录:\"use small titratable steps with frequent reassessment\" [M10#3 | 术中相关章节: Intraoperative hypotension | p.1498]。\n"
        "【决策干预（VitalDB）】：先按logged_action同类药物小步调整0.1-0.2 mL/h，目标MAP维持65-80 mmHg；2 min复评MAP/HR/SpO2，若MAP继续下降再加0.1 mL/h，若MAP>85 mmHg则回退至前一档。\n"
        "### End Example\n"
    ),
}

