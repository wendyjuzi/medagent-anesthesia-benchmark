"""Prompt/output blocks shared by GPT generation scripts."""


def build_answer_system_prompt(kind: str, include_review: bool = True) -> str:
    if kind == "miller":
        if include_review:
            return (
                "你是资深麻醉医生。仅输出中文。"
                "必须使用以下标题并按顺序输出："
                "【临床推理】、【宏观策略】、【具体干预】、【复评环节】、【原文摘录】。"
                "不得输出markdown代码块，不得输出额外说明。"
            )
        return (
            "你是资深麻醉医生。仅输出中文。"
            "必须使用以下标题并按顺序输出："
            "【临床推理】、【宏观策略】、【具体干预】、【原文摘录】。"
            "不得输出markdown代码块，不得输出额外说明。"
        )
    if include_review:
        return (
            "你是资深麻醉医生。仅输出中文。"
            "必须使用以下标题并按顺序输出："
            "【临床推理】、【宏观策略】、【具体干预】、【复评环节】。"
            "不得输出markdown代码块，不得输出额外说明。"
        )
    return (
        "你是资深麻醉医生。仅输出中文。"
        "必须使用以下标题并按顺序输出："
        "【临床推理】、【宏观策略】、【具体干预】。"
        "不得输出markdown代码块，不得输出额外说明。"
    )


def compose_final_output(
    question_text: str,
    vitaldb_text: str,
    miller_text: str,
    include_miller: bool = True,
) -> str:
    if not include_miller:
        return (
            "Q (Input Context)\n"
            f"{question_text}\n\n"
            "Answer（VitalDB版）\n"
            f"{vitaldb_text}"
        )
    return (
        "Q (Input Context)\n"
        f"{question_text}\n\n"
        "Answer（VitalDB版）\n"
        f"{vitaldb_text}\n\n"
        "Answer（Miller版）\n"
        f"{miller_text}"
    )
