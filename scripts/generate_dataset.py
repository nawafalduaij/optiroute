"""Generate data/dev_set.json (10 cases) and data/held_out_test.json (96 cases).

Held-out = 50 standard extraction cases + 46 hand-crafted hard/adversarial cases:
  12 low_lexical_overlap, 12 multi_hop, 10 arabic_orthographic_variants, 12 cache_near_duplicates
  (6 genuine paraphrase hits + 6 near-duplicate traps that the entity guard must block).
v1 had 36 cases (the first 20 standard + first 16 hard); v2 added 60 cases on 11 documents (4 new).
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
    "fleet_en": [
        "Fleet & Vehicle Policy — Al-Waha Retail Co., version 2.1, effective 2025-04-01.",
        "Eligibility: company cars are assigned to employees in Grade 10 and above and to field sales staff.",
        "Fuel: fuel is reimbursed up to SAR 1,100 per month for assigned vehicles; fuel cards must not be shared.",
        "Replacement: vehicles are replaced after 4 years or 120,000 km, whichever comes first.",
        "Traffic fines: the driver is personally responsible for traffic fines; fines paid by the company are "
        "deducted from the next salary.",
        "Accidents: any accident must be reported to the Fleet Office within 24 hours, together with the Najm "
        "accident report.",
        "Private use: limited private use is allowed within the Kingdom; cross-border trips require written "
        "approval from the Fleet Manager.",
        "Insurance: all fleet vehicles carry comprehensive insurance with a deductible of SAR 2,000 per claim, paid "
        "by the company for the first claim in a year and by the driver for any further claim in the same year.",
        "Fleet Office, Building C, ground floor. Page 1 of 1.",
    ],
    "lease_en": [
        "OFFICE LEASE AGREEMENT between Olaya Towers Real Estate Co. (\"Landlord\") and Al-Waha Retail Co. "
        "(\"Tenant\") for Unit 14-B, signed 2024-11-20.",
        "Olaya Towers Real Estate Co. — Confidential.",
        "Clause 2 – Term. The lease term is 5 years starting 2025-01-01, renewable for one further term of 3 years.",
        "Clause 3 – Rent. Annual rent is SAR 360,000, payable in four equal quarterly instalments in advance.",
        "Clause 4 – Escalation. Rent increases by 3% at the start of each lease year from year 3 onwards.",
        "Clause 5 – Security Deposit. The Tenant pays a security deposit equal to 2 months of rent, refundable "
        "within 60 days after handover.",
        "Clause 6 – Parking. The Tenant is allocated 12 parking bays; additional bays cost SAR 250 per bay per month.",
        "Clause 7 – Maintenance. The Landlord maintains the structure, lifts and central air-conditioning; the "
        "Tenant maintains the interior fit-out.",
        "Clause 9 – Early Termination. The Tenant may terminate after year 2 with 6 months' notice and a penalty "
        "of 3 months' rent.",
        "Olaya Towers Real Estate Co. — Confidential.",
    ],
    "training_ar": [
        "سياسة التدريب والتطوير لشركة الواحة للتجزئة — الإصدار ٢، مارس ٢٠٢٥.",
        "المادة ١ – الميزانية: يخصص لكل موظف مبلغ تدريب سنوي قدره ٦٬٠٠٠ ريال.",
        "المادة ٢ – الشهادات المهنية: تتحمل الشركة رسوم الشهادة المهنية الأولى بالكامل، و٥٠٪ من رسوم أي محاولة إعادة.",
        "المادة ٣ – الالتزام بالخدمة: إذا تجاوزت تكلفة البرنامج التدريبي ٢٠٬٠٠٠ ريال يلتزم الموظف بالعمل لدى "
        "الشركة مدة ١٨ شهراً بعد انتهاء البرنامج.",
        "المادة ٤ – الحضور: يجب ألا تقل نسبة حضور الموظف عن ٨٠٪ من ساعات البرنامج لاعتماد الشهادة.",
        "المادة ٦ – التعلم الإلكتروني: يُطلب من كل موظف إكمال ١٢ ساعة تعلم إلكتروني على الأقل في كل ربع سنة.",
        "المادة ٧ – الاسترداد: إذا استقال الموظف قبل انتهاء مدة الالتزام يُسترد من تكلفة البرنامج مبلغ يتناسب مع "
        "الأشهر المتبقية.",
        "مقصف المبنى الرئيسي مفتوح من الساعة ٧ صباحاً حتى ٣ عصراً.",
        "سري — للاستخدام الداخلي فقط.",
    ],
    "warehouse_ar": [
        "دليل إجراءات المستودع المركزي في الدمام — شركة الواحة للتجزئة، الإصدار ٤.",
        "الورديات: يعمل المستودع بثلاث ورديات مدة كل منها ٨ ساعات، وتبدأ الوردية الأولى الساعة ٦ صباحاً.",
        "السلامة: يُلزم جميع العاملين بارتداء الخوذة وحذاء السلامة والسترة العاكسة داخل منطقة التخزين.",
        "الرافعات الشوكية: لا يجوز تشغيل الرافعة الشوكية إلا لحاملي رخصة تشغيل سارية، والسرعة القصوى داخل "
        "الممرات ١٠ كم/ساعة.",
        "الجرد: يُجرى جرد دوري للأصناف عالية القيمة كل شهر، وجرد شامل لجميع الأصناف مرتين في السنة.",
        "الاستلام: تُفحص الشحنات الواردة خلال ٤ ساعات من وصولها، ويُرفض أي صنف تالف أو غير مطابق لأمر الشراء.",
        "التبريد: تُحفظ المواد الغذائية المبردة بين ٢ و٨ درجات مئوية، وتُسجل الحرارة كل ساعتين.",
        "السعة: تبلغ السعة التخزينية للمستودع ٤٬٥٠٠ موقع منصة نقالة.",
        "هذه الوثيقة ملك لشركة الواحة للتجزئة — صفحة ٢ من ٥.",
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

# ---------------------------------------------------------------- held-out v2: +60 cases (written and frozen
# before any model was run on them; 4 of the 11 documents are new in v2). After the first run, two reference
# fact lists were widened because they rejected correct answers (llo_ar_02: masculine المتبقي; aro_ar_08:
# "لا يمكن تمديد ... أكثر من مرة"). No system component was changed.
STANDARD_V2 = [
    case("std_en_14", "fleet_en", "standard", "What is the monthly fuel reimbursement limit for assigned vehicles?",
         1100, ["1,100"], ["up to SAR 1,100 per month"], "number"),
    case("std_en_15", "fleet_en", "standard", "Within how many hours must an accident be reported to the Fleet Office?",
         24, ["24"], ["within 24 hours"], "number"),
    case("std_en_16", "fleet_en", "standard", "After how many kilometres is a fleet vehicle replaced?",
         120000, ["120,000"], ["120,000 km"], "number"),
    case("std_en_17", "fleet_en", "standard", "Who is responsible for paying traffic fines?",
         "The driver is personally responsible.", ["driver"],
         ["the driver is personally responsible for traffic fines"]),
    case("std_en_18", "fleet_en", "standard", "Which employees are eligible for a company car?",
         "Employees in Grade 10 and above, and field sales staff.", ["Grade 10", "field sales"],
         ["Grade 10 and above and to field sales staff"]),
    case("std_en_19", "lease_en", "standard", "What is the annual rent for Unit 14-B?",
         360000, ["360,000"], ["Annual rent is SAR 360,000"], "number"),
    case("std_en_20", "lease_en", "standard", "How long is the lease term?",
         "5 years, renewable once for 3 years.", [["5 years", "five years", "5-year"]], ["The lease term is 5 years"]),
    case("std_en_21", "lease_en", "standard", "How many parking bays are allocated to the tenant?",
         12, ["12"], ["allocated 12 parking bays"], "number"),
    case("std_en_22", "lease_en", "standard", "Who maintains the central air-conditioning?",
         "The Landlord.", ["landlord"], ["The Landlord maintains the structure, lifts and central air-conditioning"]),
    case("std_en_23", "lease_en", "standard", "Within how many days is the security deposit refunded after handover?",
         60, ["60"], ["refundable within 60 days after handover"], "number"),
    case("std_en_24", "msa_en", "standard", "Which courts handle disputes under the agreement?",
         "The Riyadh commercial courts.", ["Riyadh", "commercial"], ["disputes are referred to the Riyadh commercial courts"]),
    case("std_en_25", "ap_en", "standard", "What is the status of invoice #INV-9120?",
         "Unpaid.", ["unpaid"], ["amount SAR 4,600, due 2025-10-12, status unpaid"]),
    case("std_en_26", "sec_en", "standard", "How often must passwords be rotated?",
         90, ["90"], ["rotated every 90 days"], "number"),
    case("std_en_27", "travel_en", "standard", "What is the hotel cap per night outside Riyadh, Jeddah and Dubai?",
         800, ["800"], ["up to SAR 800 per night elsewhere"], "number"),
    case("std_en_28", "sec_en", "standard", "To whom must suspected security incidents be reported?",
         "The Security Operations Centre (SOC).", [["Security Operations Centre", "Security Operations Center", "SOC"]],
         ["reported to the Security Operations Centre (SOC)"]),
    case("std_ar_08", "training_ar", "standard", "كم تبلغ ميزانية التدريب السنوية لكل موظف؟",
         6000, ["6,000"], ["مبلغ تدريب سنوي قدره ٦٬٠٠٠ ريال"], "number"),
    case("std_ar_09", "training_ar", "standard", "ما الحد الأدنى لنسبة الحضور لاعتماد الشهادة؟",
         "٨٠٪ من ساعات البرنامج", [["80%", "80"]], ["عن ٨٠٪ من ساعات البرنامج"]),
    case("std_ar_10", "training_ar", "standard", "كم ساعة تعلم إلكتروني مطلوبة في كل ربع سنة؟",
         12, ["12"], ["إكمال ١٢ ساعة"], "number"),
    case("std_ar_11", "training_ar", "standard", "ما مدة الالتزام بالخدمة بعد البرامج التدريبية المكلفة؟",
         18, ["18"], ["مدة ١٨ شهراً"], "number"),
    case("std_ar_12", "warehouse_ar", "standard", "كم مدة الوردية الواحدة في المستودع؟",
         8, ["8"], ["مدة كل منها ٨ ساعات"], "number"),
    case("std_ar_13", "warehouse_ar", "standard", "ما السرعة القصوى للرافعات الشوكية داخل الممرات؟",
         10, ["10"], ["الممرات ١٠ كم/ساعة"], "number"),
    case("std_ar_14", "warehouse_ar", "standard", "كم مرة يُجرى الجرد الشامل لجميع الأصناف؟",
         "مرتين في السنة", [["مرتين", "مرتان", "2"]], ["مرتين في السنة"]),
    case("std_ar_15", "warehouse_ar", "standard", "ما السعة التخزينية للمستودع؟",
         4500, ["4,500"], ["٤٬٥٠٠ موقع منصة نقالة"], "number"),
    case("std_ar_16", "warehouse_ar", "standard", "خلال كم ساعة تُفحص الشحنات الواردة؟",
         4, ["4"], ["خلال ٤ ساعات من وصولها"], "number"),
    case("std_ar_17", "hr_ar", "standard", "كم نسبة الأجر المدفوعة في الإجازة المرضية بعد أول ٣٠ يوماً؟",
         "ثلاثة أرباع الأجر", [["ثلاثة أرباع", "ثلاثه ارباع", "75%", "3/4"]], ["بثلاثة أرباع الأجر"]),
    case("std_ar_18", "proc_ar", "standard", "ما اللغة المعتمدة عند الاختلاف بين نصي العقد؟",
         "العربية", ["العربي"], ["ويُعتمد النص العربي عند الاختلاف"]),
    case("std_ar_19", "proc_ar", "standard", "ما قيمة أمر الشراء الملحق PO-7311؟",
         36500, ["36,500"], ["هي ٣٦٬٥٠٠ ريال سعودي"], "number"),
    case("std_ar_20", "returns_ar", "standard", "ما قيمة الخصم الذي تعادله ١٠٠ نقطة؟",
         5, ["5"], ["خصماً بقيمة ٥ ريالات"], "number"),
    case("std_ar_21", "returns_ar", "standard", "ما الحد الأدنى لقيمة الطلب ليكون شحن الإرجاع مجانياً؟",
         300, ["300"], ["تزيد قيمتها على ٣٠٠ ريال"], "number"),
    case("std_ar_22", "warehouse_ar", "standard", "ما درجة حرارة حفظ المواد الغذائية المبردة؟",
         "بين ٢ و٨ درجات مئوية", ["2", "8"], ["بين ٢ و٨ درجات مئوية"]),
]

HARD_CASES_V2 = [
    case("llo_en_04", "fleet_en", "low_lexical_overlap", "If I crash the company car, how long do I have to tell someone?",
         "Within 24 hours, to the Fleet Office, with the Najm report.", [["24 hours", "24 hrs", "one day", "1 day"]],
         ["within 24 hours"]),
    case("llo_en_05", "lease_en", "low_lexical_overlap",
         "How much cash do we have to park with the landlord up front as a guarantee?",
         "Two months of rent (SAR 60,000).", [["2 months", "two months", "60,000"]],
         ["security deposit equal to 2 months of rent"]),
    case("llo_en_06", "fleet_en", "low_lexical_overlap", "Can I drive the company car to Bahrain for the weekend?",
         "Only with written approval from the Fleet Manager.", ["approval"],
         ["cross-border trips require written approval from the Fleet Manager"]),
    case("llo_en_07", "travel_en", "low_lexical_overlap", "Can I put a family dinner on the company credit card?",
         "No. Corporate cards may not be used for personal expenses.",
         [["no", "not", "cannot", "prohibited"], "personal"], ["may not be used for personal expenses"]),
    case("llo_ar_02", "training_ar", "low_lexical_overlap", "إذا تركت الشركة بعد دورة غالية، هل علي أن أدفع شيئاً؟",
         "نعم، يُسترد من تكلفة البرنامج مبلغ يتناسب مع الأشهر المتبقية من مدة الالتزام.",
         [["يسترد", "استرداد", "يتناسب", "المتبقي"]], ["يُسترد من تكلفة البرنامج مبلغ يتناسب مع"]),
    case("llo_ar_03", "warehouse_ar", "low_lexical_overlap", "ماذا أرتدي قبل دخول منطقة الرفوف؟",
         "الخوذة وحذاء السلامة والسترة العاكسة.", ["خوذ", "حذاء"], ["بارتداء الخوذة وحذاء السلامة والسترة العاكسة"]),
    case("llo_ar_04", "hr_ar", "low_lexical_overlap", "كم يعطوني فلوس للمواصلات كل شهر؟",
         800, ["800"], ["بدل نقل ثابت قدره ٨٠٠ ريال"], "number"),
    case("llo_ar_05", "returns_ar", "low_lexical_overlap", "اشتريت شيئاً ولم أفتحه، كم لدي من الوقت لأعيده؟",
         14, ["14"], ["خلال ١٤ يوماً من تاريخ الشراء"], "number"),
    case("mh_en_04", "lease_en", "multi_hop", "Calculate the annual rent in year 3 after the escalation.",
         370800, ["370,800"], ["Annual rent is SAR 360,000", "increases by 3%"], "number"),
    case("mh_en_05", "lease_en", "multi_hop", "Calculate the early termination penalty in SAR.",
         90000, ["90,000"], ["Annual rent is SAR 360,000", "a penalty of 3 months' rent"], "number"),
    case("mh_en_06", "fleet_en", "multi_hop",
         "Calculate the total deductible the driver pays if a vehicle has 3 insurance claims in one year.",
         4000, ["4,000"], ["deductible of SAR 2,000 per claim", "by the driver for any further claim"], "number"),
    case("mh_en_07", "ap_en", "multi_hop", "Calculate the total amount of all unpaid invoices in the register.",
         17000, ["17,000"], ["amount SAR 12,400", "amount SAR 4,600"], "number"),
    case("mh_ar_02", "training_ar", "multi_hop",
         "احسب المبلغ الذي تتحمله الشركة إذا كانت رسوم الشهادة ٣٬٠٠٠ ريال واحتاج الموظف إلى محاولة إعادة واحدة.",
         4500, ["4,500"], ["رسوم الشهادة المهنية الأولى بالكامل", "و٥٠٪ من رسوم"], "number"),
    case("mh_ar_03", "hr_ar", "multi_hop",
         "احسب مجموع بدل السكن وبدل النقل الشهري لموظف راتبه الأساسي ١٢٬٠٠٠ ريال.",
         3800, ["3,800"], ["بدل سكن يعادل ٢٥٪", "٨٠٠ ريال"], "number"),
    case("mh_ar_04", "warehouse_ar", "multi_hop",
         "احسب عدد مرات الجرد في السنة (الجرد الدوري للأصناف عالية القيمة والجرد الشامل معاً).",
         14, ["14"], ["كل شهر", "مرتين في السنة"], "number"),
    case("mh_ar_05", "proc_ar", "multi_hop", "احسب مجموع قيمة أمري الشراء PO-7310 وPO-7311.",
         276500, ["276,500"], ["٢٤٠٬٠٠٠ ريال", "٣٦٬٥٠٠ ريال"], "number"),
    case("aro_ar_05", "training_ar", "arabic_orthographic_variants", "هل تدفع الشركه رسوم الشهاده المهنيه الاولي كامله؟",
         "نعم، بالكامل.", [["نعم", "بالكامل", "كامل"]], ["رسوم الشهادة المهنية الأولى بالكامل"]),
    case("aro_ar_06", "warehouse_ar", "arabic_orthographic_variants", "متي تبدا الورديه الاولي في المستودع؟",
         "الساعة ٦ صباحاً", ["6"], ["الساعة ٦ صباحاً"]),
    case("aro_ar_07", "warehouse_ar", "arabic_orthographic_variants", "كل كم ساعه تسجل درجه الحراره في التبريد؟",
         "كل ساعتين", [["ساعتين", "2"]], ["كل ساعتين"]),
    case("aro_ar_08", "hr_ar", "arabic_orthographic_variants", "هل يمكن تمديد فتره التجربه اكثر من مره؟",
         "لا، يجوز تمديدها مرة واحدة فقط.", [["مره واحده", "مرة واحدة", "لا يمكن تمديد"]], ["ويجوز تمديدها مرة واحدة"]),
    case("aro_ar_09", "returns_ar", "arabic_orthographic_variants", "ما هي المنتجات المستثناه من سياسه الارجاع؟",
         "المنتجات المخفضة ٥٠٪ أو أكثر، والمواد الغذائية، والبطاقات مسبقة الدفع.", ["غذائي", ["مسبقه", "مسبقة"]],
         ["ولا المواد الغذائية، ولا البطاقات مسبقة الدفع"]),
    case("aro_ar_10", "proc_ar", "arabic_orthographic_variants", "اين حرر عقد التوريد؟",
         "في الرياض", ["الرياض"], ["حُرر في الرياض"]),
    # cache: paraphrase hits of earlier v2 cases, and near-duplicate traps whose IDs/numbers differ
    case("cache_en_hit2", "ap_en", "cache_near_duplicates", "what's the status of invoice #INV-9120",
         "Unpaid.", ["unpaid"], ["status unpaid"], cache_of="std_en_25", expect_cache_hit=True),
    case("cache_en_trap2", "ap_en", "cache_near_duplicates", "What is the status of invoice #INV-9017?",
         "Disputed.", ["disputed"], ["status disputed"], cache_of="std_en_25", expect_cache_hit=False),
    case("cache_en_hit3", "lease_en", "cache_near_duplicates", "what's the annual rent for unit 14-B",
         360000, ["360,000"], ["Annual rent is SAR 360,000"], "number", cache_of="std_en_19", expect_cache_hit=True),
    case("cache_en_trap3", "fleet_en", "cache_near_duplicates",
         "Calculate the total deductible the driver pays if a vehicle has 4 insurance claims in one year.",
         6000, ["6,000"], ["deductible of SAR 2,000 per claim", "by the driver for any further claim"], "number",
         cache_of="mh_en_06", expect_cache_hit=False),
    case("cache_ar_hit2", "proc_ar", "cache_near_duplicates", "ما هي قيمة امر الشراء الملحق PO-7311؟",
         36500, ["36,500"], ["هي ٣٦٬٥٠٠ ريال سعودي"], "number", cache_of="std_ar_19", expect_cache_hit=True),
    case("cache_ar_trap2", "training_ar", "cache_near_duplicates",
         "احسب المبلغ الذي تتحمله الشركة إذا كانت رسوم الشهادة ٤٬٠٠٠ ريال واحتاج الموظف إلى محاولة إعادة واحدة.",
         6000, ["6,000"], ["رسوم الشهادة المهنية الأولى بالكامل", "و٥٠٪ من رسوم"], "number", cache_of="mh_ar_02",
         expect_cache_hit=False),
    case("cache_ar_hit3", "warehouse_ar", "cache_near_duplicates", "ما هي السعة التخزينية للمستودع؟",
         4500, ["4,500"], ["٤٬٥٠٠ موقع منصة نقالة"], "number", cache_of="std_ar_15", expect_cache_hit=True),
    case("cache_ar_trap3", "hr_ar", "cache_near_duplicates",
         "احسب مجموع بدل السكن وبدل النقل الشهري لموظف راتبه الأساسي ١٥٬٠٠٠ ريال.",
         4550, ["4,550"], ["بدل سكن يعادل ٢٥٪", "٨٠٠ ريال"], "number", cache_of="mh_ar_03", expect_cache_hit=False),
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
    held_out = STANDARD + STANDARD_V2 + HARD_CASES + HARD_CASES_V2
    ids = [c["task_id"] for c in held_out + DEV]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    assert len(held_out) == 96 and len(STANDARD + STANDARD_V2) == 50 and len(DEV) == 10
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
