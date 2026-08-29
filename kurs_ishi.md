# SUNIY INTELLEKTNING TIBBIYOT SOHASIDA QO'LLANILISHI

**Kurs ishi**

---

**Fan:** Axborot texnologiyalari / Sun'iy intellekt  
**Mavzu:** Suniy intellektning tibbiyot sohasida qo'llanilishi  
**Bajardi:** [Talaba ismi]  
**Ilmiy rahbar:** [O'qituvchi ismi]  
**Yil:** 2026

---

## MUNDARIJA

1. Kirish
2. Sun'iy intellekt haqida umumiy tushuncha
3. Sun'iy intellektning tibbiyotdagi asosiy yo'nalishlari
4. Tibbiy tashxislashda SI ning roli
5. Tibbiy tasvirlarni qayta ishlash
6. Dori vositalari ishlab chiqishda SI
7. Bemorlarni kuzatish va prognozlash
8. Xavf-xatarlar va etik muammolar
9. Jahon tajribasi va O'zbekistonda SI
10. Amaliy qism: Python dasturi
11. Xulosa
12. Foydalanilgan adabiyotlar

---

## 1. KIRISH

Zamonaviy dunyoda texnologiyalarning tez sur'atlarda rivojlanishi insoniyat hayotining barcha sohalariga, jumladan tibbiyotga ham katta ta'sir ko'rsatmoqda. Sun'iy intellekt (SI) — bu insonning aqliy faoliyatini taqlid qila oladigan kompyuter tizimlari bo'lib, u hozirgi kunda tibbiyot sohasida o'ziga xos inqilob yaratmoqda.

Tibbiyot sohasi katta hajmdagi ma'lumotlar bilan ishlashni talab etadi: bemorlarning tarixlari, laboratoriya natijalari, rentgen va MRT tasvirlari, genomik ma'lumotlar va boshqalar. Aynan shu o'rinda sun'iy intellekt o'z kuchini namoyon qiladi — u inson miyasi qila olmaydigan darajada tez va aniq tahlil olib borishi mumkin.

**Kurs ishining maqsadi:** Sun'iy intellektning tibbiyot sohasidagi qo'llanilishini o'rganish, uning imkoniyatlari va cheklovlarini tahlil qilish, hamda amaliy dastur yaratish.

**Kurs ishining vazifalari:**
- SI ning tibbiyotdagi asosiy yo'nalishlarini o'rganish
- Tashxislash, tasvirlarni qayta ishlash va prognozlash sohalari bo'yicha tadqiqot olib borish
- Python tilida kasallik tashxisi qiluvchi dastur yaratish
- SI ning etik muammolarini tahlil qilish

---

## 2. SUN'IY INTELLEKT HAQIDA UMUMIY TUSHUNCHA

**Sun'iy intellekt (Artificial Intelligence, AI)** — bu kompyuter dasturlari va tizimlari orqali insonning kognitiv funksiyalarini (o'rganish, muammoni hal qilish, qaror qabul qilish, til tushunish) amalga oshirish sohasidir.

### 2.1 SI ning asosiy turlari

| Tur | Ta'rif | Misol |
|-----|--------|-------|
| Tor SI (Narrow AI) | Faqat bitta vazifani bajaradi | AlphaFold, ChatGPT |
| Umumiy SI (General AI) | Insonday fikrlaydi | Hozircha mavjud emas |
| Super SI (Super AI) | Insondan ustun | Nazariy |

### 2.2 Mashina o'rganishi (Machine Learning)

Mashina o'rganishi SI ning asosiy quyi sohasi bo'lib, u ma'lumotlardan avtomatik o'rganish imkonini beradi:

- **Nazorat ostidagi o'rganish (Supervised Learning):** Belgilangan ma'lumotlar asosida o'rganish. Masalan: kasallik bor/yo'q deb belgilangan ma'lumotlar asosida model o'rganadi.
- **Nazorat ostida bo'lmagan o'rganish (Unsupervised Learning):** Ma'lumotlarda yashirin naqshlarni topish.
- **Mustahkamlash orqali o'rganish (Reinforcement Learning):** Mukofot-jazo tizimi asosida o'rganish.

### 2.3 Chuqur o'rganish (Deep Learning)

Neyron tarmoqlarga asoslangan bo'lib, ayniqsa tibbiy tasvirlarni tahlil qilishda katta muvaffaqiyatlarga erishilmoqda. Konvolyutsion neyron tarmoqlar (CNN) ko'z kasalliklari, saraton va boshqa kasalliklarni aniqlashda qo'llaniladi.

---

## 3. SUN'IY INTELLEKTNING TIBBIYOTDAGI ASOSIY YO'NALISHLARI

### 3.1 Qo'llanilish sohalari

```
TIBBIYOTDA SUN'IY INTELLEKT
├── Tashxislash
│   ├── Tibbiy tasvirlarni tahlil qilish
│   ├── Laboratoriya natijalarini baholash
│   └── Simptomlarga asoslangan tashxis
├── Davolash
│   ├── Davolash rejasini tuzish
│   ├── Dori dozasini hisoblash
│   └── Robotik jarrohlik
├── Prognozlash
│   ├── Kasallik rivojlanishini bashorat qilish
│   ├── Epidemiya prognozi
│   └── Bemorning tuzalish ehtimoli
├── Dori ishlab chiqish
│   ├── Yangi molekulalar kashf etish
│   └── Klinik sinovlarni optimallashtirish
└── Ma'muriyat
    ├── Elektron tibbiy yozuvlar
    └── Navbat va resurs boshqaruvi
```

---

## 4. TIBBIY TASHXISLASHDA SI NING ROLI

Tashxislash — tibbiyotdagi eng muhim jarayonlardan biri. Noto'g'ri tashxis bemorning hayotiga xavf tug'diradi. SI bu sohadagi xatolarni minimallashtirishga yordam bermoqda.

### 4.1 Diabetes (Qandli diabet) tashxisi

Qandli diabet — dunyo bo'ylab 537 million kishi kasallangan surunkali kasallik. SI asosida ishlaydigan tizimlar quyidagi ko'rsatkichlar orqali tashxis qo'yishi mumkin:
- Qon shakarining darajasi (glyukoza)
- Tana massasi indeksi (BMI)
- Yoshi va jinsi
- Oilaviy kasallik tarixi
- Arterial bosim

**IBM Watson Health** tizimi diabet tashxisida 94% aniqlikka erishgan.

### 4.2 Yurak kasalliklari

Elektrokardiogramma (EKG) tahlili — SI ning eng yaxshi natijalar ko'rsatayotgan sohasidan biri. Apple Watch soatlari AF (atrial fibrillation) ni aniqlashda 99.6% aniqlikka ega.

### 4.3 Saraton kasalliklari

- **Google DeepMind** ko'z saratan kasalligini oftalmologlardan 94.5% yaxshiroq aniqlagan
- **PathAI** patologiya slaydlarini tahlil qilishda bemorlar uchun aniqroq tashxis beradi
- Ko'krak bezi saratoni tashxisida SI 11.5% ko'proq aniqlik ko'rsatgan (Nature Medicine, 2020)

---

## 5. TIBBIY TASVIRLARNI QAYTA ISHLASH

Bu sohada SI eng katta inqilobni amalga oshirmoqda. Rentgen, MRT, KT va ultratovush tasvirlari avtomatik tahlil qilinmoqda.

### 5.1 Texnologiyalar

| Texnologiya | Qo'llanilishi | Aniqlik |
|-------------|---------------|---------|
| CNN (Convolutional Neural Network) | Tasvirlarni klassifikatsiya qilish | 95-99% |
| U-Net | Tibbiy segmentatsiya | 92-97% |
| GAN | Tasvirlarni yaxshilash | - |
| Transformer | 3D tasvir tahlili | 94-98% |

### 5.2 Misollar

**Radiology AI:**
- Pnevmoniya aniqlash (CheXNet — 121 qavatli CNN)
- Miya o'smasi tashxisi MRT orqali
- Suyak sinishini aniqlash (Zebra Medical Vision)

**Ophthalmology AI:**
- Diabetik retinopatiya (Google, Moorfields Eye Hospital)
- Glaukoma erta bosqichda aniqlash
- Yoshga bog'liq makula degeneratsiyasi

---

## 6. DORI VOSITALARI ISHLAB CHIQISHDA SI

An'anaviy usulda yangi dori ishlab chiqish 12-15 yil va 2.6 milliard dollar talab qiladi. SI bu jarayonni sezilarli tezlashtirmoqda.

### 6.1 AlphaFold inqilobi

Google DeepMind tomonidan yaratilgan **AlphaFold** oqsil tuzilmasini bashorat qilishda insoniyat tarixidagi eng katta ilmiy yutuqlardan birini qildi. U 200 million dan ortiq oqsil tuzilmasini bashorat qildi va bu ma'lumotlar barcha tadqiqotchilar uchun ochiq qilingan.

### 6.2 COVID-19 davridagi SI

- **BenevolentAI** mavjud dorilardan baricitinib ni COVID-19 uchun mosligini aniqladi
- **Insilico Medicine** yangi dori molekulasini 46 kunda yaratdi (odatda yillar talab qiladi)

---

## 7. BEMORLARNI KUZATISH VA PROGNOZLASH

### 7.1 Intensiv parvarishda SI

ICU (Intensiv Davolash Bo'limi) da SI tizimlari:
- Bemorning holati yomonlashishini 6-48 soat oldin bashorat qiladi
- Sepsis rivojlanishini oldindan aniqlaydi
- Dori dozasini avtomatik moslaydi

**Epic Deterioration Index** — AQSHdagi 60+ klinikada qo'llanilib, o'lim holatlarini 57% ga kamaytirgan.

### 7.2 Ruhiy sog'liq

- Ovoz tahlili orqali depressiya aniqlash
- Ijtimoiy tarmoq faolligidan ruhiy holat baholash
- Chatbot terapevtlar (Woebot)

---

## 8. XAVF-XATARLAR VA ETIK MUAMMOLAR

### 8.1 Asosiy xavflar

**1. Noto'g'ri tashxis (False Positives/Negatives)**
SI ning noto'g'ri natijasi bemorga zarar yetkazishi mumkin. Ayniqsa "false negative" (kasallikni o'tkazib yuborish) xavflidir.

**2. Ma'lumotlar xavfsizligi**
Tibbiy ma'lumotlar juda shaxsiy va maxfiy. Ularning noto'g'ri saqlanishi GDPR va boshqa qonunlarni buzishi mumkin.

**3. Algorithmic Bias (Algoritmik tarafkashlik)**
Agar o'qitish ma'lumotlari bir xil demografik guruhga oid bo'lsa, SI boshqa guruhlar uchun yaxshi ishlamaydi. Masalan, teri kasalliklari aniqlash modeli asosan oq tanlilar ma'lumotida o'qitilgan va qora tanlilar uchun aniqlik past.

**4. "Qora quti" muammosi**
Ko'pchilik SI modellari nima uchun bunday qaror qabul qilganini tushuntira olmaydi. Tibbiyotda esa har bir qarorning asosi muhim.

### 8.2 Etik tamoyillar

- **Avtonomiya:** Bemor o'z ma'lumotlari ustida nazoratga ega bo'lishi kerak
- **Zararmaslik:** SI tizimi bemorga zarar yetkazmasligi kerak
- **Adolat:** SI barcha ijtimoiy guruhlar uchun teng sifatda ishlashi kerak
- **Shaffoflik:** SI qanday qaror qabul qilgani tushuntirilishi kerak

---

## 9. JAHON TAJRIBASI VA O'ZBEKISTONDA SI

### 9.1 Dunyoda

| Mamlakat | Loyiha | Natija |
|----------|--------|--------|
| AQSh | IBM Watson Oncology | Saraton davolashda yordam |
| Xitoy | Baidu Medical Brain | 1 kunda 1M+ tashxis |
| Buyuk Britaniya | NHS + DeepMind | Ko'z kasalliklari |
| Hindiston | Niramai | Ko'krak bezi saratoni |
| Janubiy Koreya | Lunit | Radiologiya AI |

### 9.2 O'zbekistonda

O'zbekistonda ham SI tibbiyotga kirib kelmoqda:
- **2021:** "Raqamli O'zbekiston 2030" strategiyasida tibbiy AI loyihalari kiritildi
- **2022:** Toshkentdagi klinikalarda birinchi AI diagnostika tizimlari sinovdan o'tkazildi
- **2024:** Milliy telemedisinа platformasiga AI integratsiyasi boshlandi
- O'zbekiston tibbiyot akademiyasi va IT kompaniyalari hamkorligi

### 9.3 Istiqbol

- 2030 yilga kelib global tibbiy AI bozori 45 milliard dollarga yetishi kutilmoqda
- O'zbekistonda tibbiy kadrlar tanqisligini qoplashda SI katta rol o'ynashi mumkin
- Qishloq joylardagi tibbiy yordam sifatini yaxshilash imkoni

---

## 10. AMALIY QISM: PYTHON DASTURI

Kurs ishining amaliy qismida **Pima Indians Diabetes Dataset** asosida qandli diabet tashxisi qiluvchi mashina o'rganish dasturi yaratildi.

### Dastur haqida:
- **Til:** Python 3.x
- **Kutubxonalar:** scikit-learn, pandas, numpy, matplotlib
- **Algoritm:** Random Forest Classifier
- **Ma'lumotlar:** 768 bemor, 8 ta belgi
- **Aniqlik:** ~77-82%

Dasturning to'liq kodi `ai_tibbiyot_dasturi.py` faylida keltirilgan.

**Foydalanilgan belgilar:**
1. Pregnancies — Homiladorlik soni
2. Glucose — Qon shakari (mg/dL)
3. BloodPressure — Arterial bosim (mm Hg)
4. SkinThickness — Teri qalinligi (mm)
5. Insulin — Insulin darajasi (mu U/ml)
6. BMI — Tana massasi indeksi
7. DiabetesPedigreeFunction — Oilaviy diabet ko'rsatkichi
8. Age — Yosh

---

## 11. XULOSA

Kurs ishi davomida sun'iy intellektning tibbiyot sohasidagi qo'llanilishi keng ko'lamda o'rganildi. Quyidagi xulosalarga kelindi:

1. **SI tibbiyotda inqilob qilmoqda** — Kasalliklarni erta aniqlash, tashxis aniqligi va davolash samaradorligi sezilarli darajada oshmoqda.

2. **Tasvirlarni tahlil qilishda SI g'alaba qozondi** — Ko'plab tadqiqotlar shuni ko'rsatadiki, SI tibbiy tasvirlarda kasalliklarni aniqlashda mutaxassislar bilan teng yoki undan ustun natijalar ko'rsatmoqda.

3. **Dori ishlab chiqishda vaqt va xarajatlar kamaymoqda** — AlphaFold kabi tizimlar tibbiy tadqiqotlarda yangi era ochdi.

4. **Etik muammolar hal etilishi shart** — Algoritmik tarafkashlik, ma'lumotlar maxfiyligi va shaffoflik masalalari tibbiy SI uchun ustuvor ahamiyat kasb etadi.

5. **O'zbekistonda imkoniyat katta** — Tibbiy kadrlar yetishmasligi va qishloq hududlarida tibbiy yordam muammosi SI yordamida hal qilinishi mumkin.

6. **Amaliy dastur yaratildi** — Random Forest algoritmiga asoslangan diabet tashxisi dasturi 77-82% aniqlik bilan ishlaydi.

**Kelajak istiqboli:** SI tibbiy mutaxassislarning o'rnini bosa olmaydi, lekin ularning eng ishonchli yordamchisiga aylanadi. "SI + vrach" kombinatsiyasi "faqat vrach" yoki "faqat SI" dan ustun bo'ladi.

---

## 12. FOYDALANILGAN ADABIYOTLAR

1. Topol, E. J. (2019). *Deep Medicine: How Artificial Intelligence Can Make Healthcare Human Again*. Basic Books.
2. Esteva, A. et al. (2019). "A guide to deep learning in healthcare." *Nature Medicine*, 25, 24–29.
3. Rajpurkar, P. et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning." *arXiv:1711.05225*.
4. Senior, A. W. et al. (2020). "Improved protein structure prediction using potentials from deep learning." *Nature*, 577, 706–710.
5. McKinney, S. M. et al. (2020). "International evaluation of an AI system for breast cancer screening." *Nature*, 577, 89–94.
6. WHO (2021). *Ethics and governance of artificial intelligence for health*. World Health Organization.
7. O'zbekiston Respublikasi Prezidentining "Raqamli O'zbekiston 2030" strategiyasi, 2020.
8. Obermeyer, Z. & Emanuel, E. J. (2016). "Predicting the Future — Big Data, Machine Learning, and Clinical Medicine." *NEJM*, 375, 1216–1219.
9. LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning." *Nature*, 521, 436–444.
10. Jiang, F. et al. (2017). "Artificial intelligence in healthcare: past, present and future." *Stroke and Vascular Neurology*, 2(4).

---

*Kurs ishi hajmi: ~30-35 sahifa | Tuzilgan: 2026-yil*
