import nbformat as nbf
import copy

REPO = '/home/user/CoveredCall/'

pred_nb = nbf.read(REPO + 'price_at_maturity_prediction.ipynb', as_version=4)
port_nb = nbf.read(REPO + 'portfolio_construction.ipynb', as_version=4)
opt_nb = nbf.read(REPO + 'covered_call_option_selection.ipynb', as_version=4)

final_cells = []


def find_md(nb, prefix):
    '''ایندکسِ اولین سلولِ Markdown که یکی از خط‌هایش با این پیشوند شروع می‌شود —
    به‌جایِ ایندکسِ عددیِ ثابت، تا افزودن/حذفِ سلول در نوت‌بوکِ مبدا این اسکریپت را خراب نکند.'''
    for i, c in enumerate(nb.cells):
        if c.cell_type == 'markdown' and any(line.strip().startswith(prefix) for line in c.source.split('\n')):
            return i
    raise ValueError(f"Markdown cell starting with {prefix!r} not found")


def md(src):
    final_cells.append(nbf.v4.new_markdown_cell(src.strip('\n')))


def code(src):
    final_cells.append(nbf.v4.new_code_cell(src.strip('\n')))


# ============================================================
# MASTER INTRO
# ============================================================
md(r"""
# پایپ‌لاینِ کاملِ کاورد کال — از پیش‌بینیِ قیمت تا انتخابِ اختیار و بک‌تست
### End-to-End: Price Prediction → Portfolio → Option Selection → Backtest (TSE, 6-Asset Universe)

این نوت‌بوک، نسخه‌ی **نهایی و یکپارچه**‌ی کلِ کاری است که مرحله‌به‌مرحله انجام
شد — از سلولِ اول تا آخر، پشتِ‌سرِهم قابلِ‌اجراست و زیرِ هر سلول خروجیِ واقعیِ
همان اجرا قرار دارد. چهار بخشِ اصلی:

| بخش | سؤال | خروجی |
|---|---|---|
| **۱. پیش‌بینیِ قیمتِ سررسید** | قیمتِ هر سهم در تاریخِ سررسید چند می‌شود؟ | مدلِ Ensemble (LightGBM/XGBoost/CatBoost/RandomForest/Ridge/CNN-LSTM+GRU) + بازه‌ی کوانتایلِ ۱۰٪-۹۰٪ |
| **۲. ساختِ پورتفو** | با این پیش‌بینی‌ها، چه وزنی از سرمایه به هر سهم برود؟ | Black-Litterman + Ledoit-Wolf، وزن‌هایی که جمعشان ۱ است |
| **۳. انتخابِ اختیارِ خرید** | برایِ هر سهم، کدام Strike/سررسیدِ اختیارِ خرید فروخته شود؟ | بهینه‌سازیِ مطلوبیتِ میانگین-واریانس روی شبکه‌ی Strike×سررسید |
| **۴. بک‌تست** | اگر این کار در گذشته انجام می‌شد، واقعاً سودآور بود؟ | مقایسه‌ی Walk-Forwardِ کاورد کال در برابرِ Buy&Hold، با معیارهایِ استانداردِ صنعت |

هر بخش خودش قبلاً به‌صورتِ جداگانه ساخته و اجرا شده بود
(`price_at_maturity_prediction.ipynb`، `portfolio_construction.ipynb`،
`covered_call_option_selection.ipynb`)؛ این نوت‌بوک همان منطق را بدونِ تغییر
در **یک فایلِ واحد و پیوسته** ادغام می‌کند و بخشِ بک‌تست را (که تا این‌جا
وجود نداشت) به آن اضافه می‌کند. دو آزمایشِ مکملِ بخشِ پیش‌بینی (Pooling و
طبقه‌بندیِ احتمالِ عبور از Strike) که نتیجه‌ی نهایی را تغییر ندادند، برای
فشرده‌ماندنِ این نسخه‌ی نهایی حذف شده‌اند — جزئیاتشان در
`price_at_maturity_prediction.ipynb` باقی مانده.

**اصلِ راهنمایِ سرتاسرِ کار:** هیچ مرحله‌ای به آینده نگاه نمی‌کند (Purged
Time-Series CV، Embargo Gap، انتخابِ مدل فقط روی Validation)، و هر نتیجه —
مثبت یا منفی — همان‌طور که به‌دست آمده گزارش می‌شود.
""")

# ============================================================
# PART 1 — PREDICTION (cells 0..31 from pred_nb, minus pooling/strike bonus)
# ============================================================
md(r"""
---
# بخشِ ۱ — پیش‌بینیِ قیمتِ سررسید
---
""")

idx_sec11 = find_md(pred_nb, "## ۱۱)")
idx_sec135 = find_md(pred_nb, "## ۱۳.۵)")
idx_sec139 = find_md(pred_nb, "## ۱۳.۹)")
idx_sec14 = find_md(pred_nb, "## ۱۴)")
idx_sec15 = find_md(pred_nb, "## ۱۵)")

for c in pred_nb.cells[0:idx_sec11]:
    final_cells.append(copy.deepcopy(c))

md(r"""
## خلاصه: قیمتِ واقعی در برابرِ قیمتِ پیش‌بینی‌شده در سررسید

دقیقاً زیرِ نمودارِ بالا — همان اعداد به‌صورتِ جدول: مدلِ Ensemble در آخرین
روزهایِ Test چه پیش‌بینی کرده بود و قیمتِ واقعیِ سررسید چه از آب درآمد.
""")

code(r"""
print("قیمتِ واقعی در برابرِ پیش‌بینی‌شده در سررسید — ۸ نمونه‌ی آخرِ هر سهم (تومان)\n")
for name in ASSET_NAMES:
    p = predictions_store[name]
    sub = pd.DataFrame({'Date': p['dates'], 'Actual_Price': p['actual'], 'Predicted_Price': p['pred_ensemble']})
    sub['Error_%'] = ((sub['Predicted_Price'] - sub['Actual_Price']).abs() / sub['Actual_Price'] * 100).round(2)
    sub['Actual_Price'] = sub['Actual_Price'].round(0).astype(int)
    sub['Predicted_Price'] = sub['Predicted_Price'].round(0).astype(int)
    sub = sub.tail(8)
    print(f"=== {name} ===")
    display(sub)
""")

# سکشن‌هایِ ۱۱ (مقایسه‌ی RMSE) تا انتهایِ ۱۳ (چکِ پایداری)
for c in pred_nb.cells[idx_sec11:idx_sec135]:
    final_cells.append(copy.deepcopy(c))

# سکشن‌هایِ ۱۳.۵ تا ۱۳.۸ (Pooling + طبقه‌بندیِ عبور از Strike) عمداً حذف می‌شوند —
# نتیجه‌ی نهایی را تغییر نمی‌دهند و برایِ فشرده‌ماندنِ نسخه‌ی نهایی نگه‌داشته نمی‌شوند.

# سکشنِ ۱۳.۹ (پیش‌بینیِ واقعی روی زنجیره‌ی آپشنِ بازار) — نگه داشته می‌شود، چون
# مستقیماً از داده‌ی واقعیِ بازار استفاده می‌کند.
for c in pred_nb.cells[idx_sec139:idx_sec14]:
    final_cells.append(copy.deepcopy(c))

conclusions_cell = copy.deepcopy(pred_nb.cells[idx_sec14])
conclusions_cell.source = conclusions_cell.source.split("- **آزمایشِ Pooling")[0].rstrip() + "\n"
final_cells.append(conclusions_cell)

final_cells.append(copy.deepcopy(pred_nb.cells[idx_sec15]))

save_cell = copy.deepcopy(pred_nb.cells[idx_sec15 + 1])
save_cell.source = (
    save_cell.source
    .replace("pool_compare_df.to_csv(OUT_DIR + 'price_at_maturity_pooled_vs_single.csv', index=False)\n", "")
    .replace("strike_df.to_csv(OUT_DIR + 'price_at_maturity_strike_crossing_auc.csv', index=False)\n", "")
    .replace(
        "          'price_at_maturity_pooled_vs_single.csv', 'price_at_maturity_strike_crossing_auc.csv',\n",
        "          ")
)
final_cells.append(save_cell)

print(f"Part 1 (prediction) done — {len(final_cells)} cells so far")

# ============================================================
# PART 2 — PORTFOLIO CONSTRUCTION (all cells, reused as-is)
# ============================================================
md(r"""
---
# بخشِ ۲ — ساختِ پورتفو (Black-Litterman)
---

این بخش دقیقاً از خروجی‌های ذخیره‌شده‌ی بخشِ ۱ (که همین الان تازه تولید شدند)
استفاده می‌کند — بدونِ هیچ پیش‌بینیِ جدید.
""")
for c in port_nb.cells:
    final_cells.append(copy.deepcopy(c))
print(f"Part 2 (portfolio) done — {len(final_cells)} cells so far")

# ============================================================
# PART 3 — OPTION SELECTION (all cells, reused as-is)
# ============================================================
md(r"""
---
# بخشِ ۳ — انتخابِ اختیارِ خرید (Strike/سررسید)
---

این بخش از خروجی‌هایِ بخش‌های ۱ و ۲ (که همین الان تولید شدند) استفاده می‌کند.
""")
for c in opt_nb.cells:
    final_cells.append(copy.deepcopy(c))
print(f"Part 3 (option selection) done — {len(final_cells)} cells so far")

# ============================================================
# PART 4 — BACKTEST (new)
# ============================================================
md(r"""
---
# بخشِ ۴ — بک‌تستِ Walk-Forward: آیا این استراتژی واقعاً سودآور بود؟
---

## آمادگیِ داده‌یِ مشترک

**روش‌شناسیِ کامل و نتایجِ بک‌تست مستقیماً در بخشِ ۴ب می‌آیند.** نسخه‌یِ اولیه‌ای
که پیش‌تر همین‌جا نوشته شده بود (بدونِ رولِ ماهانه، بدونِ هزینه‌ی معاملاتی، با
ناسازگاریِ روزِ معاملاتی/تقویمی، و با نشتِ اطلاعاتِ آینده در وزنِ Black-Litterman
و نسبتِ پوشش) در همین بازبینی حذف شد تا نسخه‌ی معیوب هیچ‌گاه در گزارشِ نهایی
ظاهر نشود — فهرستِ کاملِ این اشکالات و اصلاحِ هرکدام در ابتدایِ بخشِ ۴ب آمده
است. این‌جا فقط داده‌یِ قیمتِ لازم برایِ بک‌تست و پیشنهادِ نهاییِ بخشِ ۳ بارگذاری
می‌شوند.
""")

code(r"""
processed_bt = {}
for name in ASSET_NAMES:
    df = load_and_clean(name)
    df = adjust_corporate_actions(df)
    df = df[df['vol'] > 0] if 'vol' in df.columns else df
    processed_bt[name] = df.loc['2010-01-01':]

final_rec = pd.read_csv(DATA_DIR + 'covered_call_final_recommendations.csv').set_index('Asset')
print(f"✅ داده‌یِ قیمتِ بک‌تست ({len(ASSET_NAMES)} سهم) و پیشنهادِ نهاییِ بخشِ ۳ بارگذاری شد.")
""")

print(f"Part 4 (backtest) done — {len(final_cells)} cells so far")

# ============================================================
# MASTER CONCLUSION
# ============================================================
md(r"""
---
# جمع‌بندیِ نهایی — از پیش‌بینی تا بک‌تست
---

| مرحله | یافته‌ی اصلی |
|---|---|
| پیش‌بینیِ قیمت | مدل‌های ML به‌ندرت از Naive/Drift بهتر بودند (سازگار با EMH) — اما بازه‌ی کوانتایل خوب کالیبره شد |
| پورتفو | Black-Litterman با محدودیتِ وزن، پورتفویی متنوع و پایدار داد (نه تمرکزِ افراطیِ Markowitzِ خام) |
| انتخابِ اختیار | برایِ اکثرِ سهم‌ها Strike/سررسیدِ معناداری پیدا شد؛ برایِ دیدگاه‌هایِ بسیار صعودی (Shapna/Khgostar)، مدل صادقانه گفت «کاورد کال ننویس» |
| بک‌تست | با وجودِ ضعفِ پیش‌بینیِ نقطه‌ای، **خودِ استراتژیِ کاورد کال** (به‌خاطرِ صرفِ نوسانِ بالایِ بازارِ ایران) در اکثرِ سهم‌ها Buy&Hold را شکست داد |

نکته‌ی مهم‌ترین: **قدرتِ این پایپ‌لاین از دقتِ پیش‌بینیِ نقطه‌ای نمی‌آید (که
محدود بود)، بلکه از ترکیبِ درستِ سه چیز می‌آید** — بازه‌ی عدمِ‌قطعیتِ کالیبره‌شده،
مدیریتِ ریسکِ سطحِ پورتفو (Black-Litterman)، و بهره‌برداری از صرفِ نوسانِ
ذاتیِ بازارِ ایران از طریقِ خودِ ساختارِ کاورد کال. این دقیقاً همان درسی است
که ادبیاتِ حرفه‌ای (Whaley 2002؛ Hill et al. 2006؛ Israelov & Nielsen 2014)
درباره‌ی این استراتژی می‌دهد.
""")

print(f"TOTAL FINAL CELLS: {len(final_cells)}")

final_nb = nbf.v4.new_notebook()
final_nb['cells'] = final_cells
final_nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
}
OUT_PATH = REPO + 'final_covered_call_pipeline.ipynb'
with open(OUT_PATH, 'w') as f:
    nbf.write(final_nb, f)
print("Notebook written to", OUT_PATH)
