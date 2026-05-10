"""Shared text mappings for QA generation."""

from typing import Dict

CN_TERM_MAP: Dict[str, str] = {
    "M": "男性",
    "F": "女性",
    "Male": "男性",
    "Female": "女性",
    "General surgery": "普外科",
    "Thoracic Surgery": "胸外科",
    "Thoracic surgery": "胸外科",
    "Cardiac Surgery": "心脏外科",
    "Neurosurgery": "神经外科",
    "Intraoperative": "术中",
    "relative timestamp": "相对时间",
    "Unknown": "暂缺",
    "Unknown surgery": "手术名称暂缺",
    "Advanced gastric cancer": "进展期胃癌",
    "Aortic aneurysm": "主动脉瘤",
    "Aortic aneurys": "主动脉瘤",
    "Hepatocellular carcinoma": "肝细胞癌",
    "Hepatic": "肝脏",
    "Liver": "肝脏",
    "Stomach": "胃",
    "Vascular": "血管外科",
    "Subtotal gastrectomy": "胃次全切除术",
    "Liver segmentectomy": "肝段切除术",
    "Aneurysmal repair": "动脉瘤修补术",
    "Lobectomy": "肺叶切除术",
    "VATS": "胸腔镜手术",
}

SURGERY_CN_MAP: Dict[str, str] = {
    "Subtotal gastrectomy": "胃次全切除术",
    "Liver segmentectomy": "肝段切除术",
    "Aneurysmal repair": "动脉瘤修补术",
    "Lobectomy": "肺叶切除术",
    "Video-assisted thoracoscopic surgery": "胸腔镜手术",
}
