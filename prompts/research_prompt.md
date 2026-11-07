# RESEARCH PROMPT — “ล็อกดาวน์ = วันหยุดบางส่วน” (Holiday‑Equivalent Stoppage, HES)
## พยากรณ์มูลค่าการส่งออกรายเดือนของไทยผ่านช่วงช็อกโควิด‑19 ด้วย Prophet โดยถ่ายโอนผลการหยุดงานจากสงกรานต์และตรุษจีน

> **วิธีใช้:** คัดลอกทั้งไฟล์ไปวางใน AI agent ที่รันโค้ดได้ (เช่น Claude Code) แนบไฟล์ `hesi_forecast.py` และไฟล์ข้อมูลส่งออกรายเดือน (CSV: `month,value`) แล้วสั่งให้เริ่มที่ Phase 0
> **สัดส่วนงาน:** Academic 45% : Code 55% (รายละเอียดในหัวข้อ 3)

---

## 0) บทบาทของคุณ

คุณคือทีมวิจัยสองบทบาทในคนเดียว

1. **Academic lead (นักเศรษฐมิติอนุกรมเวลา):** รับผิดชอบคำถามวิจัย สมมติฐาน การระบุค่าพารามิเตอร์ (identification) การอนุมานทางสถิติ และการเขียน paper
2. **Code lead (ML/forecasting engineer):** รับผิดชอบ pipeline ที่ reproducible ไม่มี data leakage มี unit test และรันซ้ำได้ด้วยคำสั่งเดียว

ทำงานแบบนักวิจัยที่ซื่อตรง รายงานผลที่ไม่เข้าข้างสมมติฐานด้วย **ห้ามแต่งตัวเลข ห้ามแต่งเอกสารอ้างอิง** ถ้าไม่แน่ใจข้อเท็จจริงใด ให้ระบุว่าไม่แน่ใจและบอกวิธีตรวจสอบ

---

## 1) บริบทและแนวคิดตั้งต้น

- **ข้อมูลหลัก:** มูลค่าการส่งออกรายเดือนของไทย (USD) จากแหล่ง public ตั้งแต่ปี 2010 ถึงเดือนล่าสุด
- **ปัญหา:** ช่วงโควิด‑19 (2020–2022) ยอดส่งออกตกและเด้งกลับผิดปกติ อนุกรมเวลาจึง “พัง” ในสามด้าน คือ trend และ seasonality ถูกบิด, lag feature ปนเปื้อน และ tree model (LightGBM) extrapolate ไม่ได้
- **แนวคิด:** ล็อกดาวน์คือ “การหยุดทำงาน” แบบเดียวกับวันหยุดยาว สงกรานต์คือการหยุดงานในประเทศ ส่วนตรุษจีนคือคู่ค้าจีนหยุด วันหยุดเกิดซ้ำทุกปี จึงใช้สอนโมเดลได้ว่า “หยุดงาน 1 วัน ยอดส่งออกหายเท่าไร” แล้วนำผลนั้นไปใช้ในช่วงโควิด

### บทเรียนจาก simulation study ที่ทำไว้แล้ว (ต้องนำมาใช้ ห้ามทำผิดซ้ำ)

1. **ห้ามเปิด holiday effect เต็มเดือนแบบ binary** เพราะวันล็อกดาวน์ไม่ได้หยุดเท่าวันสงกรานต์ ผลจึง overshoot ช่วงที่เปิด และพลาดช่วงที่ไม่ได้เปิด
2. **ใช้ “ปริมาณ” (dose) แทน on/off** ด้วยดัชนี HESI ซึ่งบอกว่าวันหนึ่งในช่วงโควิดเทียบเท่าวันหยุดกี่ส่วน (สูตรอยู่ในหัวข้อ 5)
3. **ใช้ shared regressor ตัวเดียวต่อช่องทาง** ห้ามสร้าง holiday ชื่อใหม่ว่า “covid” ใน Prophet เพราะ Prophet จะสร้างพารามิเตอร์ใหม่ที่ไม่มีข้อมูลให้เรียนก่อนเกิดเหตุ ผลคือไม่มีการถ่ายโอนเลย
4. **ข้อมูลรายวันระบุผลวันหยุดได้แม่นกว่ารายเดือนมาก** ในการจำลอง โมเดลรายวันประมาณ β ได้ตรงค่าจริง (−0.80) แต่โมเดลรายเดือนได้ราว −0.46 ถ้ามีแต่ข้อมูลรายเดือนต้องใช้ aggregation correction (AC)
5. **LightGBM ถ่ายโอน dose ไม่ได้** threshold ที่เรียนได้บนตัวแปร 0/1 อยู่ที่ 1e‑35 ทำให้ dose เพียงเล็กน้อยก็กลายเป็นวันหยุดเต็มวัน
6. **Stringency ≠ การหยุดงาน** ดัชนี OxCGRT ของจีนสูง 0.6–0.8 ตลอดสามปีของนโยบาย zero‑COVID ขณะที่โรงงานส่วนใหญ่ยังเดิน ถ้าใช้ค่าดิบจะทำลายโมเดล
7. **ในช่วงปกติ Prophet ไม่ใช่โมเดลที่ดีที่สุด** (SARIMAX และ LightGBM แม่นกว่า) ต้องรายงานตรงไปตรงมา

---

## 2) คำถามวิจัยและสมมติฐาน

| RQ | คำถาม | สมมติฐาน (ทดสอบได้) |
|---|---|---|
| RQ1 Transfer | ผลการหยุดงานที่เรียนจากวันหยุด ถ่ายโอนไปพยากรณ์ช่วงล็อกดาวน์ได้หรือไม่ | H1: Prophet + shared regressor + HESI ให้ error ช่วง 2020–2021 ต่ำกว่า (a) ไม่ใส่โควิด (b) binary ทั้งเดือน (c) stringency ดิบ อย่างมีนัยสำคัญ (DM‑HLN, p<0.05) |
| RQ2 Dose | ความแม่นยำไวต่อการกำหนด dose แค่ไหน และปรับเทียบอย่างไร | H2: error เป็นรูปตัว U ตามตัวคูณ λ การปรับ λ จาก 1–2 เดือนแรกของล็อกดาวน์ (HESI‑cal) ทำได้ใกล้ oracle |
| RQ3 Frequency | train รายวันดีกว่ารายเดือนหรือไม่ | H3: β̂ จากข้อมูลรายวันมี bias และ variance ต่ำกว่า โมเดลรายเดือนถูกลดทอน (attenuation) และ AC ลด bias ได้ |
| RQ4 Model class | tree model ถ่ายโอนผลได้หรือไม่ | H4: tree เรียน dose‑response แบบขั้นบันได จึงแย่กว่าโมเดลเชิงเส้นในช่วงช็อก |
| RQ5 Post‑shock | การจัดการช่วงโควิดช่วยพยากรณ์ปี 2023 เป็นต้นไปหรือไม่ | H5: HESI หรือ mask ลด error เทียบกับการไม่จัดการ และ stringency ดิบทำให้แย่ลง |

---

## 3) สัดส่วนงาน Academic : Code = 45 : 55

| ส่วน | % | งาน | ผลลัพธ์ที่ต้องส่ง |
|---|---|---|---|
| A1 ทฤษฎีและ literature | 10 | calendar effects, intervention analysis, Prophet, COVID ใน time series, tree forecasting | related work และตารางเปรียบเทียบแนวทาง |
| A2 การออกแบบวิจัย | 15 | RQ/H, identification, evaluation design, กฎกัน leakage | research design 2–3 หน้า |
| A3 ตีความและอภิปราย | 10 | ผล, ข้อจำกัด, ข้อเสนอเชิงปฏิบัติ | ส่วน discussion |
| A4 เขียน paper | 10 | LaTeX ไทย/อังกฤษ | `paper.pdf` |
| C1 Data pipeline | 15 | ดาวน์โหลด ทำความสะอาด จัด vintage | `data/` + `data_manifest.csv` |
| C2 Feature engineering | 10 | ปฏิทิน, HESI, regressors, AC | `features.py` + tests |
| C3 Models | 15 | Prophet D/M, SARIMAX, LightGBM, SNaive (+ ETS/อื่น ๆ ถ้าเวลาพอ) | `models/` |
| C4 Evaluation & statistics | 10 | rolling origin, metrics, DM‑HLN, ablation, placebo | `results/*.csv` |
| C5 Reproducibility | 5 | Makefile, pinned versions, seeds, README | `make all` รันผ่าน |

**สัดส่วนเนื้อหาใน paper.pdf:** เชิงวิชาการประมาณ 80% และโค้ด (ภาคผนวก) ประมาณ 20%

---

## 4) ข้อมูล

### 4.1 ตัวแปรตาม
- **แหล่ง:** สำนักงานนโยบายและยุทธศาสตร์การค้า (สนค.) กระทรวงพาณิชย์ (ระบบสถิติการค้า), ธนาคารแห่งประเทศไทย (สถิติการค้าระหว่างประเทศ) และกรมศุลกากร
- เก็บทั้ง **ส่งออกรวม** และ **ส่งออกไม่รวมทองคำ** ทองคำผันผวนสูงมากในปี 2020 ให้ใช้แบบไม่รวมทองคำเป็นหลัก และใช้แบบรวมเป็น robustness
- บันทึก **vintage** (วันที่ดาวน์โหลด, URL, sha256) เพราะตัวเลขย้อนหลังถูกปรับปรุงได้
- ถ้าเข้าถึง **ข้อมูลรายวันหรือรายสัปดาห์** ได้ (เช่น ข้อมูลใบขนสินค้า) ให้ทำ Prophet รายวันเป็นแบบหลัก ถ้าไม่มี ให้ใช้รายเดือนร่วมกับ AC

### 4.2 ปฏิทินวันหยุดไทย
- ใช้ `holidays.Thailand(language="en_US")` เป็นวันหยุดที่ **หยุดจริง** รวมวันชดเชยและวันหยุดพิเศษ และตัด “National Children's Day” ออก
- จุดตรวจสำคัญ: ปี 2020 สงกรานต์ถูกเลื่อน และมีวันหยุดชดเชย 27 ก.ค., 4 ก.ย. และ 7 ก.ย. 2020 ให้ตรวจกับมติ ครม. อีกครั้ง

### 4.3 ตรุษจีน
- ใช้ `holidays.China(language="en_US")` หาวันแรก “Chinese New Year (Spring Festival)” แล้วกำหนดหน้าต่าง [−1, +6] วัน
- sensitivity: [−3, +10]

### 4.4 ตัวขับเคลื่อนช่วงโควิด
- **OxCGRT Stringency Index** (THA, CHN; national): `github.com/OxCGRT/covid-policy-dataset`
- **Google COVID‑19 Community Mobility Reports** (Thailand, หมวด workplaces, 15 ก.พ. 2020 – ส.ค./ต.ค. 2022) ใช้ mirror บน GitHub ได้
- **ช่องทางคู่ค้าจีน:** ใช้ episode index ของการหยุดอุตสาหกรรม (ล็อกดาวน์ ม.ค.–มี.ค. 2020 และเซี่ยงไฮ้ เม.ย.–พ.ค. 2022) **ไม่ใช่ stringency ดิบ** ถ้าหาได้ ให้ลอง proxy ที่ดีกว่า เช่น PMI ด้านการผลิต, ปริมาณตู้สินค้าท่าเรือ หรือข้อมูล NO₂ จากดาวเทียม

### 4.5 ตัวแปรเสริม (robustness)
อัตราแลกเปลี่ยน, PMI โลก, ราคาน้ำมัน ต้องใช้ค่าที่รู้ ณ วันพยากรณ์ หรือทำเป็น scenario เท่านั้น

---

## 5) การสร้างตัวแปร (หัวใจของงาน)

```
H_d = 1 ถ้าเป็นวันหยุดราชการไทย,   C_d = 1 ถ้าอยู่ในหน้าต่างตรุษจีน
drop_d = max(0, −workplaces_d / 100)
HESI_d = clip( smooth7_excl_holidays(drop_d) / median(drop บนวันหยุดที่ตรงกับวันทำงาน), 0, 1 )

x^D_d = H_d + (1 − H_d) · min(1, λ · HESI_d)        # ช่องทางในประเทศ
x^P_d = C_d + (1 − C_d) · PHESI_d                    # ช่องทางคู่ค้า (จีน)

log y_d = g(t) + s_year(t) + s_week(t) + β_D x^D_d + β_P x^P_d + ε_d
```

- ใช้ `add_regressor(..., standardize=False)` แบบ additive บน `log(y)` แล้วแปลงกลับด้วย Duan smearing
- เหตุผล: multiplicative mode เมื่อผลใหญ่ซ้อนกัน (สงกรานต์ + วันอาทิตย์) อาจทำให้ค่าพยากรณ์ติดลบ
- **ข้อมูลรายเดือน:** `y_m = log(ยอดรวม / จำนวนวัน)` และ regressors = ค่าเฉลี่ยรายวันของ x^D, x^P ในเดือน + สัดส่วนวันเสาร์‑อาทิตย์ (WK)
- **Aggregation correction (AC):** หา β_d ที่ทำให้ `log mean_d exp(β_d x_d)` ตรงกับ `β_M · x̄_m` ในช่วง train แล้วพยากรณ์ด้วย `log mean_d exp(β_d x_d)`
- **Variants ที่ต้องเปรียบเทียบ:**
  - Off — ไม่ใส่โควิด
  - Binary — เปิดทั้งเดือนเมื่อ stringency เฉลี่ยของเดือน ≥ 0.70
  - Stringency ดิบ
  - HESI (λ=1)
  - HESI‑cal — λ̂ จาก 1–2 เดือนแรกที่ประกาศตัวเลขแล้ว
  - Sep — ใส่ตัวขับเคลื่อนโควิดเป็น regressor แยก มีสัมประสิทธิ์ของตัวเอง
  - Mask — ตัดช่วงโควิดออกจากข้อมูล train
- **Sensitivity:** แยกค่าอ้างอิงของเทศกาลใหญ่ (สงกรานต์/ปีใหม่ ลดลงราว 0.64–0.70) ออกจากวันหยุดเดี่ยว (ราว 0.47), เปลี่ยน smoothing window และใช้ trailing window แทน centered (สำหรับ real‑time)

---

## 6) โมเดล

| รหัส | โมเดล | สเปก |
|---|---|---|
| SNaive | Seasonal naive | ค่าเดือนเดียวกันของปีล่าสุด |
| P‑D | Prophet รายวัน (ถ้ามีข้อมูลรายวัน) | log y, yearly=10, weekly=3, regressors x^D, x^P |
| P‑M | Prophet รายเดือน | log(avg/day), yearly=5, n_changepoints=15, regressors x̄^D, x̄^P, WK |
| P‑M+AC | Prophet รายเดือน + aggregation correction | เหมือน P‑M แล้วแปลงผลด้วย β_d |
| SARIMAX | Airline (0,1,1)(0,1,1)₁₂ + exog | exog เดียวกับ P‑M |
| LGBM | Hybrid: linear log‑trend + LightGBM บน residual | features: month, x̄^D, x̄^P, WK (lag ≥ horizon เท่านั้น) |
| (เสริม) | ETS, TBATS, N‑BEATS/TFT | ถ้าเวลาพอ |

กำหนด hyperparameter ล่วงหน้า ถ้าจะ tune ต้องใช้ time‑series CV ภายในช่วง train เท่านั้น

---

## 7) การออกแบบการประเมิน

- **E0 ช่วงปกติ:** origin 2017‑12 → พยากรณ์ 2018‑01..2019‑12
- **E1 ช่วงช็อก (scenario‑conditional / nowcasting):** origin 2019‑12 → พยากรณ์ 2020‑01..2021‑12 โดยใส่ตัวขับเคลื่อนที่สังเกตได้ในอนาคต **ต้องติดป้ายว่าเป็น conditional ไม่ใช่ ex‑ante**
- **E2 หลังช็อก:** origin 2022‑12 → พยากรณ์ 2023‑01 ถึงเดือนล่าสุด
- **Rolling origin:** ทุก 6 เดือน, h = 1..12 และรายงานแยกตาม horizon
- **Metrics:** MAPE, sMAPE, MASE (scale = seasonal naive in‑sample), RMSE และ interval coverage 80/95% ถ้าทำ uncertainty
- **สถิติ:** Diebold–Mariano พร้อม Harvey–Leybourne–Newbold correction (Newey–West lag h−1), Model Confidence Set (ทางเลือก) และรายงาน effect size พร้อม CI เสมอ
- **Ablation:** ตัด HESI, ตัด CNY channel, เปลี่ยนหน้าต่าง CNY, เปลี่ยนค่าอ้างอิง mobility, เปลี่ยน threshold ของ binary
- **Placebo:** ใส่ “โควิดปลอม” ในปี 2017 ซึ่งไม่ควรช่วยให้ error ลดลง

---

## 8) กฎกัน data leakage (บังคับทุกข้อ)

1. ทุกอย่างที่ fit (β, scaling, smoothing reference, λ̂) ใช้ข้อมูลที่ ≤ origin เท่านั้น
2. centered smoothing ใช้ได้ในงานแบบ ex‑post ส่วน real‑time ต้องใช้ trailing window และรายงานทั้งสองแบบ
3. HESI‑cal ใช้เฉพาะเดือนที่ประกาศตัวเลขแล้ว (ตัวเลขส่งออกไทยประกาศราว 3–4 สัปดาห์หลังสิ้นเดือน)
4. การใช้ตัวขับเคลื่อนในอนาคตถือเป็น conditional forecast และต้องแยกตารางรายงาน
5. ถ้าทำได้ ใช้ real‑time vintages แทนข้อมูลที่ถูก revise ภายหลัง
6. lag feature ของ tree model ต้อง ≥ horizon
7. เขียน `test_no_leakage.py` ตรวจว่าไม่มีแถวหลัง origin หลุดเข้า training

---

## 9) Simulation study (validation ก่อนใช้ข้อมูลจริง)

- รัน `python hesi_forecast.py simulate --reps 30` (DGP รายวันที่รู้ค่าจริง ใช้ปฏิทินและตัวขับเคลื่อนโควิดจริง)
- **S1 ideal:** κ = 1 (mobility = การหยุดผลิตจริง)
- **S2 realistic:** κ = 0.5 + demand shock ปี 2020 และ rebound ปี 2021
- ตรวจสามเรื่อง: (1) วิธีการ recover β ได้ (2) จุดต่ำสุดของ λ‑sweep อยู่ที่ κ (3) daily ดีกว่า monthly และ tree ล้มเหลวตามทฤษฎี
- **ขยายต่อ:**
  - S3 วันหยุดมีผลไม่เท่ากัน (สงกรานต์หนักกว่าวันหยุดเดี่ยว)
  - S4 proxy bias เปลี่ยนตามเวลา (WFH เพิ่มขึ้น)
  - S5 ไม่มีการถ่ายโอน (โควิดไม่เกี่ยวกับการหยุดงาน) เพื่อทดสอบว่าวิธีไม่ทำร้ายโมเดล

---

## 10) โครงสร้างโปรเจกต์

```
hes-export/
├── README.md
├── requirements.txt              # pin versions
├── Makefile                      # make data | sim | backtest | paper | all
├── data/{raw,interim,processed}/ + data_manifest.csv (source, url, date, sha256)
├── src/hes/
│   ├── calendar.py  drivers.py  features.py  simulate.py  evaluation.py  report.py
│   └── models/{prophet_log.py, sarimax.py, lgbm_hybrid.py, baselines.py}
├── scripts/01_download.py 02_features.py 03_backtest.py 04_simulation.py 05_figures.py
├── tests/test_calendar.py test_hesi.py test_linear_trick.py test_no_leakage.py
├── results/{metrics.csv, forecasts.csv, coefficients.csv, figures/}
└── paper/{paper.tex, refs.bib, figures/} → paper.pdf
```

**Unit tests ขั้นต่ำ:**
1. สงกรานต์ 2020 ไม่อยู่ในเดือนเมษายน
2. HESI อยู่ในช่วง [0, 1]
3. ค่าพยากรณ์ของ Prophet = base0 + Σ coef·x แบบตรงเป๊ะ
4. ไม่มี training row หลัง origin

---

## 11) paper.pdf

- **ภาษา:** ไทยพร้อม Abstract อังกฤษ เขียนด้วย XeLaTeX + ฟอนต์ไทย (Sarabun / TH Sarabun New) และ `\XeTeXlinebreaklocale "th"`
- **ความยาว:** 12–20 หน้า
- **ตัวเลขทุกตัวในข้อความและตาราง** ต้อง generate จากไฟล์ผล (macros) ห้ามพิมพ์มือ

**โครงเรื่อง**
1. บทคัดย่อ (ไทย/อังกฤษ) + คำสำคัญ
2. บทนำ: ปัญหา, ช่องว่างความรู้, contributions 3–5 ข้อ
3. งานที่เกี่ยวข้อง
   - calendar/moving‑holiday effects: Bell & Hillmer 1983; Findley et al. 1998; Lin & Liu 2003
   - intervention analysis: Box & Tiao 1975
   - Prophet: Taylor & Letham 2018 และเอกสาร “Handling Shocks”
   - COVID ใน time series: Lenza & Primiceri 2022; Foroni et al. 2022
   - tree forecasting: Ke et al. 2017; Makridakis et al. 2022; Januschowski et al. 2022
   - ข้อมูลนโยบายและ mobility: Hale et al. 2021; Google
4. ระเบียบวิธี: แบบจำลอง, HESI, propositions (transfer, Jensen/aggregation, identification รายเดือน, tree step function), AC
5. ข้อมูล
6. การออกแบบการทดลอง (simulation + ข้อมูลจริง)
7. ผล: E0/E1/E2, λ‑sweep, ความแม่นยำของ β̂, DM tests, ablation, placebo
8. อภิปราย: เมื่อใดวิธีนี้ใช้ได้หรือไม่ได้, ข้อจำกัดอย่างน้อย 5 ข้อ, ข้อเสนอเชิงปฏิบัติ
9. สรุป
10. Reproducibility statement และ data availability
11. เอกสารอ้างอิง (ตรวจ DOI ทุกรายการ)
12. ภาคผนวก: โค้ดหลัก (HESI, regressor, AC) และตารางเพิ่มเติม

**รูปบังคับ:**
1. กรอบแนวคิด
2. ข้อมูลพร้อมช่วงโควิด
3. ตัวขับเคลื่อนและ HESI
4. forecast paths
5. λ‑sweep
6. boxplot ของ β̂
7. dot plot ของ E2

**ตารางบังคับ:**
1. นิยามโมเดล
2. ผล E0/E1/E2 (mean ± se และ DM p‑value)
3. β̂
4. λ̂

---

## 12) Definition of done

- [ ] `make all` รันผ่านบนเครื่องใหม่
- [ ] ตัวเลขทุกตัวใน paper มาจากไฟล์ผล
- [ ] การเปรียบเทียบหลักทุกคู่มี DM‑HLN test
- [ ] รายงานกรณีที่โมเดลที่เสนอแพ้ด้วย
- [ ] แยกชัดว่าอะไรเป็น conditional forecast
- [ ] ตรวจ citation ทุกตัว (DOI/ISBN)
- [ ] ระบุข้อจำกัดอย่างน้อย 5 ข้อ

---

## 13) Pitfalls ที่ต้องเลี่ยงและต้องเขียนถึงใน paper

- holiday ชื่อใหม่ “covid” ใน Prophet ไม่เกิดการถ่ายโอน
- binary ทั้งเดือนทำให้ overshoot หรือพลาดช่วงที่ไม่ได้เปิด
- stringency ไม่ใช่ตัววัดการหยุดงาน (กรณีจีนภายใต้ zero‑COVID)
- mobility ไม่ใช่การผลิตเพื่อส่งออก (คนทำงานจากบ้านแต่โรงงานยังเดิน) จึงมักมี pass‑through κ < 1
- demand shock ของโลกไม่ใช่การหยุดงาน HESI จึงอธิบายไม่ได้ และ λ̂ ที่ calibrate จะดูดซับ demand ไปด้วย
- ทองคำทำให้เกิด outlier
- หน่วย USD กับ THB ให้ผลต่างกัน
- ในข้อมูลรายเดือน สงกรานต์ชนกับฤดูกาลเดือนเมษายน (collinearity)
- Prophet multiplicative mode กับผลขนาดใหญ่ซ้อนกันอาจให้ค่าติดลบ
- LightGBM threshold 1e‑35 บนตัวแปร 0/1

---

## 14) รูปแบบการรายงานของคุณ

ทำงานเป็นเฟส:

**Phase 0** Plan → **1** Data → **2** Features → **3** Simulation → **4** Backtest → **5** Statistics → **6** Paper → **7** Self‑review

จบแต่ละเฟสให้รายงานสั้น ๆ ในรูปแบบนี้:
1. สิ่งที่ทำ
2. ไฟล์ที่ได้
3. ผลสำคัญ (ตัวเลข)
4. ปัญหาและการตัดสินใจ
5. ขั้นต่อไป

ใน Phase 7 ให้วิจารณ์งานตัวเองแบบ reviewer อย่างน้อย 5 ประเด็น แล้วแก้ก่อนส่ง
