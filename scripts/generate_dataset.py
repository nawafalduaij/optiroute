"""Generate data/dev_set.json (10 cases) and data/held_out_test.json (36 cases).

Held-out = 20 standard extraction cases + 16 hand-crafted hard/adversarial cases:
  4 low_lexical_overlap, 4 multi_hop, 4 arabic_orthographic_variants, 4 cache_near_duplicates
  (2 genuine paraphrase hits + 2 near-duplicate traps that the entity guard must block).
All contexts are synthetic enterprise documents (fictional companies). Deterministic output.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ---------------------------------------------------------------- documents (one chunk per paragraph)
DOCS: dict[str, list[str]] = {
    "msa_en": [
        "MASTER SERVICES AGREEMENT between Najd Cloud Services LLC (\"Provider\") and Al-Waha Retail Co. "
        "(\"Customer\"), effective 2025-03-01.",
        "Najd Cloud Services LLC — Confidential. All rights reserved.",
        "Section 2 – Fees. The Customer shall pay a monthly platform fee of SAR 48,000, invoiced on the first "
        "business day of each month. Payment is due within 30 days of the invoice date.",
        "Section 3 – Late Payment. Overdue amounts accrue a late charge of 1.5% per month on the outstanding balance.",
        "Section 4 – Support. Priority-1 incidents are acknowledged within 15 minutes, 24/7. Priority-2 incidents "
        "are acknowledged within 4 business hours.",
        "Section 5 – Data Residency. All customer data shall be stored in data centres located within the Kingdom "
        "of Saudi Arabia (Riyadh and Dammam regions).",
        "Section 6 – Service Credits. If monthly availability falls below 99.9%, the Customer receives a service "
        "credit of 10% of the monthly platform fee. If availability falls below 99.0%, the credit increases to 25% "
        "of the monthly platform fee.",
        "Section 7 – Termination. Either party may terminate this Agreement for convenience with 90 days' written "
        "notice.",
        "Section 8 – Governing Law. This Agreement is governed by the laws of the Kingdom of Saudi Arabia; "
        "disputes are referred to the Riyadh commercial courts.",
        "Najd Cloud Services LLC — Confidential. All rights reserved.",
    ],
    "ap_en": [
        "Accounts Payable Register — Q3 2025, Al-Waha Retail Co. Prepared by the Finance Shared Services team.",
        "Invoice #INV-8841 from Gulf Packaging Est.: amount SAR 12,400, due 2025-09-15, status unpaid. "
        "Invoice #INV-8842 from Gulf Packaging Est.: amount SAR 9,850, due 2025-09-30, status paid.",
        "Invoice #INV-9017 from Red Sea Logistics: amount SAR 31,200, due 2025-10-05, status disputed because of a "
        "quantity mismatch against purchase order PO-5520.",
        "VAT: all supplier invoices are subject to 15% VAT, which is already included in the amounts listed in this "
        "register unless stated otherwise.",
        "Early payment discount: suppliers on the Preferred list (Gulf Packaging Est., Najm Office Supplies) grant "
        "a 2% discount on the invoice amount if paid within 10 days of the invoice date.",
        "Invoice #INV-9120 from Najm Office Supplies: amount SAR 4,600, due 2025-10-12, status unpaid.",
        "The AP team sits on the 3rd floor of Olaya Tower and can be reached at extension 4410.",
        "Page 1 of 1 — Confidential.",
    ],
    "sec_en": [
        "Information Security Policy — Al-Waha Retail Co., version 4.2, approved by the CISO.",
        "Password Policy: passwords must be at least 14 characters long and rotated every 90 days. Reuse of the "
        "last 5 passwords is prohibited.",
        "Multi-Factor Authentication: MFA is mandatory for all remote access and for every privileged account.",
        "Incident Reporting: suspected security incidents must be reported to the Security Operations Centre (SOC) "
        "within 1 hour of detection.",
        "Data Classification: data is classified as Public, Internal, Confidential or Restricted. Restricted data "
        "may not be stored on removable media.",
        "Laptop Encryption: all company laptops must use full-disk encryption (BitLocker or FileVault).",
        "Access Reviews: managers must review user access rights every quarter; privileged access is reviewed "
        "monthly.",
        "Cafeteria opening hours are 7:30 to 15:00 on working days.",
    ],
    "travel_en": [
        "Travel & Expense Policy — Al-Waha Retail Co., effective 2025-01-01.",
        "Per diem: employees travelling domestically receive a per diem of SAR 450 per day; the international "
        "travel per diem is SAR 900 per day.",
        "Flights: economy class is used for flights under 6 hours; business class is permitted for flights of 6 "
        "hours or longer for Grade 12 and above.",
        "Hotel cap: hotel accommodation is reimbursed up to SAR 1,200 per night in Riyadh, Jeddah and Dubai, and "
        "up to SAR 800 per night elsewhere.",
        "Claims must be submitted within 30 days of return, together with original receipts.",
        "Mileage: use of a personal car for business is reimbursed at SAR 0.75 per kilometre.",
        "Corporate cards may not be used for personal expenses under any circumstances.",
        "This document is property of Al-Waha Retail Co. Page 1 of 2.",
    ],
    "hr_ar": [
        "سياسة الموارد البشرية لشركة الواحة للتجزئة — الإصدار ٣، يناير ٢٠٢٥.",
        "سري للغاية — جميع الحقوق محفوظة لشركة الواحة للتجزئة.",
        "المادة ٤ – الإجازة السنوية: يستحق الموظف إجازة سنوية مدفوعة الأجر مدتها ٢١ يوماً، تزداد إلى ٣٠ يوماً "
        "بعد إتمام خمس سنوات من الخدمة المتصلة.",
        "المادة ٥ – الإجازة المرضية: تُدفع الإجازة المرضية بالأجر الكامل لأول ٣٠ يوماً، ثم بثلاثة أرباع الأجر "
        "للـ ٦٠ يوماً التالية.",
        "المادة ٦ – بدل السكن: يُصرف للموظف بدل سكن يعادل ٢٥٪ من الراتب الأساسي، ويُدفع شهرياً مع الراتب.",
        "المادة ٧ – بدل النقل: يُصرف بدل نقل ثابت قدره ٨٠٠ ريال شهرياً لجميع الموظفين.",
        "المادة ٩ – العمل عن بُعد: يجوز للموظف العمل عن بُعد بحد أقصى يومين في الأسبوع بعد موافقة المدير المباشر.",
        "المادة ١٢ – فترة التجربة: تكون فترة التجربة ٩٠ يوماً، ويجوز تمديدها مرة واحدة بموافقة كتابية.",
        "سري للغاية — جميع الحقوق محفوظة لشركة الواحة للتجزئة.",
    ],
    "proc_ar": [
        "عقد توريد بين شركة الواحة للتجزئة ومؤسسة الخليج للتغليف، حُرر في الرياض بتاريخ ٢٠٢٥/٠٢/١٠.",
        "البند ٣ – قيمة العقد: القيمة الإجمالية لأمر الشراء PO-7310 هي ٢٤٠٬٠٠٠ ريال سعودي، وقيمة أمر الشراء "
        "الملحق PO-7311 هي ٣٦٬٥٠٠ ريال سعودي، غير شاملة ضريبة القيمة المضافة.",
        "البند ٥ – مدة التوريد: يلتزم المورد بتسليم الطلبات خلال ١٤ يوم عمل من تاريخ أمر الشراء.",
        "البند ٨ – غرامة التأخير: في حال تأخر التسليم تُفرض غرامة قدرها ٠٫٥٪ من قيمة أمر الشراء PO-7310 عن كل "
        "يوم تأخير، بحد أقصى ١٠٪ من قيمة العقد.",
        "البند ١٠ – الضمان: يضمن المورد المنتجات ضد عيوب التصنيع لمدة ١٢ شهراً من تاريخ الاستلام.",
        "البند ١١ – الدفع: تُسدد الفواتير خلال ٤٥ يوماً من تاريخ استلام الفاتورة المعتمدة.",
        "البند ١٥ – اللغة: حُرر هذا العقد باللغتين العربية والإنجليزية، ويُعتمد النص العربي عند الاختلاف.",
        "هذه الوثيقة ملك لشركة الواحة للتجزئة — صفحة ١ من ٣.",
    ],
    "returns_ar": [
        "سياسة الإرجاع والاستبدال لمتاجر الواحة — سارية من ١ مارس ٢٠٢٥.",
        "يحق للعميل إرجاع المنتجات غير المستخدمة خلال ١٤ يوماً من تاريخ الشراء مع تقديم الفاتورة الأصلية.",
        "تُسترد المبالغ إلى وسيلة الدفع الأصلية خلال ٧ أيام عمل من استلام المنتج المرتجع.",
        "لا تشمل سياسة الإرجاع المنتجات المخفضة بنسبة ٥٠٪ أو أكثر، ولا المواد الغذائية، ولا البطاقات مسبقة الدفع.",
        "رسوم شحن الإرجاع: مجانية للطلبات التي تزيد قيمتها على ٣٠٠ ريال، و٢٥ ريالاً للطلبات الأقل من ذلك.",
        "برنامج الولاء: يحصل العميل على نقطة واحدة لكل ١٠ ريالات يتم إنفاقها، وتعادل كل ١٠٠ نقطة خصماً بقيمة ٥ ريالات.",
        "جميع الحقوق محفوظة لمتاجر الواحة.",
    ],
}

DOC_LANG = {k: ("ar" if k.endswith("_ar") else "en") for k in DOCS}


def case(task_id, doc, category, query, expected, facts, evidence, answer_type="text", schema=None,
         cache_of=None, expect_cache_hit=None):
    return {
        "task_id": task_id, "language": DOC_LANG[doc] if category != "cross" else "en", "category": category,
        "query": query, "context": DOCS[doc], "expected_answer": expected, "answer_type": answer_type,
        "required_facts": facts, "evidence": evidence, "schema": schema, "cache_of": cache_of,
        "expect_cache_hit": expect_cache_hit, "doc": doc,
    }


# ---------------------------------------------------------------- held-out: 20 standard cases
STANDARD = [
    case("std_en_01", "msa_en", "standard", "What is the monthly platform fee under the agreement?",
         48000, ["48,000"], ["SAR 48,000"], "number"),
    case("std_en_02", "msa_en", "standard", "How much written notice is required to terminate for convenience?",
         "90 days' written notice", [["90 days", "ninety days"]], ["90 days' written notice"]),
    case("std_en_03", "msa_en", "standard", "Where must customer data be stored?",
         "In data centres within Saudi Arabia (Riyadh and Dammam regions).",
         [["Saudi Arabia", "KSA"], "Riyadh", "Dammam"], ["Riyadh and Dammam regions"]),
    case("std_en_04", "ap_en", "standard", "What is the amount of invoice #INV-8841?",
         12400, ["12,400"], ["amount SAR 12,400"], "number"),
    case("std_en_05", "ap_en", "standard", "Why is invoice #INV-9017 disputed?",
         "Quantity mismatch against purchase order PO-5520.", ["quantity mismatch", "PO-5520"],
         ["quantity mismatch against purchase order PO-5520"]),
    case("std_en_06", "ap_en", "standard",
         "Give the invoice ID, supplier and status for invoice #INV-9017 as JSON.",
         {"invoice_id": "INV-9017", "supplier": "Red Sea Logistics", "status": "disputed"},
         ["Red Sea Logistics", "disputed"], ["Invoice #INV-9017 from Red Sea Logistics"], "json",
         schema={"invoice_id": "string", "supplier": "string", "status": "string"}),
    case("std_en_07", "sec_en", "standard", "What is the minimum password length?",
         14, ["14"], ["at least 14 characters"], "number"),
    case("std_en_08", "sec_en", "standard", "How often is privileged access reviewed?",
         "Monthly.", ["monthly"], ["privileged access is reviewed monthly"]),
    case("std_en_09", "sec_en", "standard", "Which full-disk encryption tools are approved for company laptops?",
         "BitLocker or FileVault.", ["BitLocker", "FileVault"], ["BitLocker or FileVault"]),
    case("std_en_10", "sec_en", "standard", "When is multi-factor authentication mandatory?",
         "For all remote access and for every privileged account.", ["remote access", "privileged"],
         ["MFA is mandatory for all remote access"]),
    case("std_en_11", "travel_en", "standard", "What is the domestic per diem?",
         450, ["450"], ["per diem of SAR 450"], "number"),
    case("std_en_12", "travel_en", "standard", "What is the hotel cap per night in Jeddah?",
         1200, ["1,200"], ["up to SAR 1,200 per night in Riyadh, Jeddah and Dubai"], "number"),
    case("std_en_13", "travel_en", "standard", "Which cabin class should be booked for a 4-hour flight?",
         "Economy class.", ["economy"], ["economy class is used for flights under 6 hours"]),
    case("std_ar_01", "hr_ar", "standard", "كم مدة الإجازة السنوية للموظف؟",
         21, ["21"], ["مدتها ٢١ يوماً"], "number"),
    case("std_ar_02", "hr_ar", "standard", "كم يبلغ بدل النقل الشهري؟",
         800, ["800"], ["بدل نقل ثابت قدره ٨٠٠ ريال"], "number"),
    case("std_ar_03", "hr_ar", "standard", "ما مدة فترة التجربة؟",
         90, ["90"], ["فترة التجربة ٩٠ يوماً"], "number"),
    case("std_ar_04", "proc_ar", "standard", "ما القيمة الإجمالية لأمر الشراء PO-7310؟",
         240000, ["240,000"], ["هي ٢٤٠٬٠٠٠ ريال سعودي"], "number"),
    case("std_ar_05", "proc_ar", "standard", "خلال كم يوماً يجب سداد الفواتير؟",
         45, ["45"], ["خلال ٤٥ يوماً"], "number"),
    case("std_ar_06", "returns_ar", "standard", "كم يوماً لدى العميل لإرجاع المنتجات؟",
         14, ["14"], ["خلال ١٤ يوماً من تاريخ الشراء"], "number"),
    case("std_ar_07", "returns_ar", "standard", "خلال كم يوم عمل يتم استرداد المبلغ؟",
         7, ["7"], ["خلال ٧ أيام عمل"], "number"),
]

# ---------------------------------------------------------------- held-out: 16 hard / adversarial cases
HARD_CASES = [
    # Low lexical overlap: query paraphrases the target sentence with (almost) no shared keywords.
    case("llo_en_01", "msa_en", "low_lexical_overlap", "How fast must the vendor respond to a critical outage?",
         "Within 15 minutes, 24/7.", [["15 minutes", "fifteen minutes"]], ["acknowledged within 15 minutes"]),
    case("llo_en_02", "sec_en", "low_lexical_overlap", "Am I allowed to copy top-secret files onto a USB stick?",
         "No. Restricted data may not be stored on removable media.",
         [["no", "not", "prohibited", "cannot"], "restricted"], ["may not be stored on removable media"]),
    case("llo_en_03", "travel_en", "low_lexical_overlap",
         "What's the deadline for filing my expense paperwork after I get back from a trip?",
         "Within 30 days of return, with original receipts.", [["30 days", "thirty days"]],
         ["within 30 days of return"]),
    case("llo_ar_01", "proc_ar", "low_lexical_overlap", "ما الحد الأعلى للعقوبة المالية عند التأخر؟",
         "١٠٪ من قيمة العقد", [["10%", "24,000"]], ["بحد أقصى ١٠٪ من قيمة العقد"]),
    # Multi-hop: needs facts from two separate chunks plus arithmetic.
    case("mh_en_01", "msa_en", "multi_hop",
         "Calculate the service credit in SAR for a month in which availability was 98.7%.",
         12000, ["12,000"], ["SAR 48,000", "the credit increases to 25%"], "number"),
    case("mh_en_02", "ap_en", "multi_hop",
         "Calculate the amount payable for invoice #INV-9120 if it is paid within the early payment window.",
         {"invoice_id": "INV-9120", "amount_payable": 4508}, ["4,508"],
         ["amount SAR 4,600", "2% discount"], "json",
         schema={"invoice_id": "string", "amount_payable": "number"}),
    case("mh_en_03", "travel_en", "multi_hop",
         "Calculate the total reimbursable amount for a 3-night international trip to Dubai: per diem for 4 days "
         "plus hotel at the cap for 3 nights.",
         7200, ["7,200"], ["SAR 900 per day", "up to SAR 1,200 per night in Riyadh, Jeddah and Dubai"], "number"),
    case("mh_ar_01", "proc_ar", "multi_hop", "احسب قيمة غرامة التأخير إذا تأخر المورد ٦ أيام في تسليم أمر الشراء PO-7310.",
         7200, ["7,200"], ["٢٤٠٬٠٠٠ ريال", "غرامة قدرها ٠٫٥٪"], "number"),
    # Arabic orthographic / clitic / numeral variants (missing hamza, ه for ة, ي for ى, attached clitics).
    case("aro_ar_01", "hr_ar", "arabic_orthographic_variants", "ما نسبه بدل السكن بالنسبه للراتب الاساسي؟",
         "٢٥٪ من الراتب الأساسي", [["25%", "25"]], ["بدل سكن يعادل ٢٥٪"]),
    case("aro_ar_02", "hr_ar", "arabic_orthographic_variants", "هل يستحق الموظف 30 يوما اجازه سنويه بعد 5 سنوات خدمه؟",
         "نعم، تزداد الإجازة السنوية إلى ٣٠ يوماً بعد خمس سنوات.", [["نعم", "يستحق"], "30"],
         ["تزداد إلى ٣٠ يوماً"]),
    case("aro_ar_03", "proc_ar", "arabic_orthographic_variants", "ما مده الضمان علي المنتجات ضد عيوب التصنيع؟",
         12, ["12"], ["لمدة ١٢ شهراً"], "number"),
    case("aro_ar_04", "returns_ar", "arabic_orthographic_variants", "هل يمكن ارجاع المنتجات المخفضه بنسبه ٦٠٪؟",
         "لا، المنتجات المخفضة بنسبة ٥٠٪ أو أكثر غير مشمولة بسياسة الإرجاع.",
         [["لا", "غير مشموله", "لا تشمل", "لا يمكن"]], ["المخفضة بنسبة ٥٠٪ أو أكثر"]),
    # Cache: genuine paraphrases (should hit Tier-0) and near-duplicate traps (must be blocked by the guard).
    case("cache_en_hit", "ap_en", "cache_near_duplicates", "what's the amount of invoice #INV-8841",
         12400, ["12,400"], ["amount SAR 12,400"], "number", cache_of="std_en_04", expect_cache_hit=True),
    case("cache_en_trap", "ap_en", "cache_near_duplicates", "What is the amount of invoice #INV-8842?",
         9850, ["9,850"], ["amount SAR 9,850"], "number", cache_of="std_en_04", expect_cache_hit=False),
    case("cache_ar_hit", "proc_ar", "cache_near_duplicates", "ما هي القيمة الاجمالية لامر الشراء PO-7310؟",
         240000, ["240,000"], ["هي ٢٤٠٬٠٠٠ ريال سعودي"], "number", cache_of="std_ar_04", expect_cache_hit=True),
    case("cache_ar_trap", "proc_ar", "cache_near_duplicates", "احسب قيمة غرامة التأخير إذا تأخر المورد ٨ أيام في تسليم أمر الشراء PO-7310.",
         9600, ["9,600"], ["٢٤٠٬٠٠٠ ريال", "غرامة قدرها ٠٫٥٪"], "number", cache_of="mh_ar_01",
         expect_cache_hit=False),
]

# ---------------------------------------------------------------- dev set (threshold calibration only)
DEV = [
    case("dev_en_01", "msa_en", "standard", "What late charge applies to overdue amounts?",
         "1.5% per month", ["1.5%"], ["late charge of 1.5% per month"]),
    case("dev_en_02", "msa_en", "standard", "How quickly are Priority-2 incidents acknowledged?",
         "Within 4 business hours.", [["4 business hours", "four business hours"]], ["within 4 business hours"]),
    case("dev_en_03", "sec_en", "low_lexical_overlap", "How soon do I need to flag a suspected hack?",
         "Within 1 hour of detection.", [["1 hour", "one hour", "an hour", "60 minutes"]], ["within 1 hour"]),
    case("dev_en_04", "travel_en", "multi_hop", "Calculate the mileage reimbursement for driving 120 km on business.",
         90, ["90"], ["SAR 0.75 per kilometre"], "number"),
    case("dev_en_05", "ap_en", "standard", "When is invoice #INV-9120 due?",
         "2025-10-12", ["2025-10-12"], ["due 2025-10-12"]),
    case("dev_ar_01", "hr_ar", "standard", "ما مدة الإجازة المرضية بالأجر الكامل؟",
         30, ["30"], ["بالأجر الكامل لأول ٣٠ يوماً"], "number"),
    case("dev_ar_02", "hr_ar", "standard", "كم يوماً يُسمح بالعمل عن بُعد في الأسبوع؟",
         "يومين", [["يومين", "يومان", "2"]], ["بحد أقصى يومين في الأسبوع"]),
    case("dev_ar_03", "returns_ar", "standard", "كم رسوم شحن الإرجاع لطلب قيمته ٢٠٠ ريال؟",
         25, ["25"], ["و٢٥ ريالاً"], "number"),
    case("dev_ar_04", "proc_ar", "standard", "خلال كم يوم عمل يجب على المورد تسليم الطلبات؟",
         14, ["14"], ["خلال ١٤ يوم عمل"], "number"),
    case("dev_ar_05", "returns_ar", "multi_hop", "كم نقطة يحصل عليها العميل عند إنفاق ٤٠٠ ريال، وما قيمة خصمها؟",
         "40 نقطة", ["40"], ["نقطة واحدة لكل ١٠ ريالات"]),
]


def main() -> None:
    held_out = STANDARD + HARD_CASES
    ids = [c["task_id"] for c in held_out + DEV]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    assert len(held_out) == 36 and len(STANDARD) == 20 and len(DEV) == 10
    for c in held_out + DEV:
        for ev in c["evidence"]:
            assert any(ev in chunk for chunk in c["context"]), f"{c['task_id']}: evidence not in context: {ev}"
    DATA.mkdir(exist_ok=True)
    (DATA / "held_out_test.json").write_text(json.dumps(held_out, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / "dev_set.json").write_text(json.dumps(DEV, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / "distractor_pool.json").write_text(
        json.dumps({k: v for k, v in DOCS.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ar = sum(c["language"] == "ar" for c in held_out)
    by_cat: dict[str, int] = {}
    for c in held_out:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
    print(f"held_out={len(held_out)} (en={len(held_out) - n_ar}, ar={n_ar}) dev={len(DEV)} categories={by_cat}")


if __name__ == "__main__":
    main()
