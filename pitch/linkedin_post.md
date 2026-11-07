# LinkedIn post: Lockdowns as Partial Holidays

Image: `docs/assets/linkedin_poster.png` (1440 x 1800, 4:5). Every number below is in the paper or computed from
the data in the repository; export values in the study are simulated, which the post says.

---

## English

14 April 2020 should have been Songkran.

Instead, Thailand postponed the holiday and went to work under lockdown. That day, Thai workplaces emptied 66% as much as they do on a real public holiday.

So I asked: what if a lockdown is just a partial holiday?

Every forecasting model hit the same wall in 2020. No parameter estimated before COVID could describe a lockdown, because there had never been one. But Thailand runs a natural experiment every year: Songkran, public holidays, and Chinese New Year at our largest trading partner all stop work for days at a time.

The idea, in three steps:
1️⃣ Learn the work-stoppage effect from ten years of holidays, with one shared regressor per channel in Prophet
2️⃣ Measure each lockdown day in holiday units: the Holiday-Equivalent Stoppage Index (HESI), built from Google Mobility
3️⃣ Forecast the shock with the same effect. No new parameter.

In a 30-replication Monte Carlo study (simulated exports, real holiday calendar and real COVID data):
📉 When the dose is right, forecast error for 2020-21 falls from 26.0% to 7.8% MAPE
📐 Theory and simulation agree: monthly models shrink the elasticity to −0.54, against −0.55 predicted
🌳 LightGBM cannot do this. Any positive dose becomes a full holiday
⚖️ When mobility overstates the stoppage two-fold, the gain is no longer significant, and after the shock, plain SARIMAX does as well

That last point matters to me. HES is a tool for measurable stoppage shocks, not a universal accuracy fix, and the paper says so.

Everything is open:
📄 Paper (read online): https://thanatchon36.github.io/lockdowns-as-partial-holidays/paper/
💻 Code: https://github.com/thanatchon36/lockdowns-as-partial-holidays
🤗 Data: https://huggingface.co/datasets/thanatchon36/lockdowns-as-partial-holidays
🇹🇭 Thai version included · DOI 10.5281/zenodo.24036954

Next step: official Thai export statistics. If you work with Thai trade data, I would love to compare notes.

Independent personal research; views are my own.

#Forecasting #TimeSeries #EconomicResearch #Thailand #COVID19 #OpenScience #DataScience

---

## ภาษาไทย

14 เมษายน 2563 ควรเป็นวันสงกรานต์

แต่ปีนั้นไทยเลื่อนวันหยุดออกไป และผู้คนต้องทำงานภายใต้ล็อกดาวน์ วันนั้นที่ทำงานทั่วประเทศว่างลงถึง 66% ของวันหยุดราชการจริง

ผมเลยตั้งคำถามว่า ถ้าล็อกดาวน์ก็คือ "วันหยุดบางส่วน" ล่ะ?

ปี 2563 โมเดลพยากรณ์ทุกตัวเจอกำแพงเดียวกัน คือไม่มีพารามิเตอร์ไหนที่ประมาณก่อนโควิดจะอธิบายล็อกดาวน์ได้ เพราะไม่เคยเกิดมาก่อน แต่ไทยมีการทดลองธรรมชาติทุกปีอยู่แล้ว ทั้งสงกรานต์ วันหยุดราชการ และตรุษจีนของคู่ค้าใหญ่ที่สุดของเรา ซึ่งล้วนทำให้การทำงานหยุดลงหลายวัน

แนวคิดมีสามขั้น
1️⃣ เรียนรู้ผลของการหยุดงานจากวันหยุดสิบปี ด้วยตัวแปรร่วมตัวเดียวต่อช่องทางใน Prophet
2️⃣ วัดวันล็อกดาวน์แต่ละวันเป็น "หน่วยวันหยุด" ด้วยดัชนี HESI ที่คำนวณจาก Google Mobility
3️⃣ พยากรณ์ช่วงช็อกด้วยผลเดิม ไม่ต้องเพิ่มพารามิเตอร์ใหม่เลย

จากการศึกษาแบบ Monte Carlo 30 รอบ (ยอดส่งออกเป็นข้อมูลจำลอง แต่ปฏิทินวันหยุดและข้อมูลโควิดเป็นข้อมูลจริง)
📉 เมื่อปริมาณ dose ถูกต้อง ค่าคลาดเคลื่อนการพยากรณ์ปี 2563-64 ลดจาก 26.0% เหลือ 7.8% (MAPE)
📐 ทฤษฎีตรงกับผลจำลอง โมเดลรายเดือนหดค่า elasticity เหลือ −0.54 เทียบกับที่ทำนายไว้ −0.55
🌳 LightGBM ทำแบบนี้ไม่ได้ dose เท่าไรก็กลายเป็นวันหยุดเต็มวัน
⚖️ ถ้า mobility ประเมินการหยุดงานเกินจริงสองเท่า ผลดีขึ้นจะไม่มีนัยสำคัญ และหลังช็อก SARIMAX ธรรมดาก็ทำได้ดีพอกัน

ข้อสุดท้ายสำคัญสำหรับผม HES เป็นเครื่องมือสำหรับช็อกที่วัดการหยุดงานได้ ไม่ใช่ยาวิเศษที่ทำให้แม่นขึ้นทุกกรณี และในเปเปอร์ก็เขียนไว้ตรง ๆ

ทุกอย่างเปิดให้ใช้
📄 เปเปอร์ (อ่านออนไลน์): https://thanatchon36.github.io/lockdowns-as-partial-holidays/paper/
💻 โค้ด: https://github.com/thanatchon36/lockdowns-as-partial-holidays
🤗 ข้อมูล: https://huggingface.co/datasets/thanatchon36/lockdowns-as-partial-holidays
🇹🇭 มีฉบับภาษาไทย · DOI 10.5281/zenodo.24036954

ขั้นต่อไปคือข้อมูลส่งออกทางการของไทย ใครทำงานกับข้อมูลการค้าไทย ยินดีแลกเปลี่ยนมากครับ

งานวิจัยส่วนตัว ความเห็นเป็นของผมเอง

#Forecasting #TimeSeries #EconomicResearch #Thailand #COVID19 #OpenScience #DataScience
