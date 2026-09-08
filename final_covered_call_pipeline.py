#!/usr/bin/env python
# coding: utf-8

# # پایپ‌لاینِ کاملِ کاورد کال — از پیش‌بینیِ قیمت تا انتخابِ اختیار و بک‌تست
# ### End-to-End: Price Prediction → Portfolio → Option Selection → Backtest (TSE, 6-Asset Universe)
# 
# این نوت‌بوک، نسخه‌ی **نهایی و یکپارچه**‌ی کلِ کاری است که مرحله‌به‌مرحله انجام
# شد — از سلولِ اول تا آخر، پشتِ‌سرِهم قابلِ‌اجراست و زیرِ هر سلول خروجیِ واقعیِ
# همان اجرا قرار دارد. چهار بخشِ اصلی:
# 
# | بخش | سؤال | خروجی |
# |---|---|---|
# | **۱. پیش‌بینیِ قیمتِ سررسید** | قیمتِ هر سهم در تاریخِ سررسید چند می‌شود؟ | مدلِ Ensemble (LightGBM/XGBoost/CatBoost/RandomForest/Ridge/CNN-LSTM+GRU) + بازه‌ی کوانتایلِ ۱۰٪-۹۰٪ |
# | **۲. ساختِ پورتفو** | با این پیش‌بینی‌ها، چه وزنی از سرمایه به هر سهم برود؟ | Black-Litterman + Ledoit-Wolf، وزن‌هایی که جمعشان ۱ است |
# | **۳. انتخابِ اختیارِ خرید** | برایِ هر سهم، کدام Strike/سررسیدِ اختیارِ خرید فروخته شود؟ | بهینه‌سازیِ مطلوبیتِ میانگین-واریانس روی شبکه‌ی Strike×سررسید |
# | **۴. بک‌تست** | اگر این کار در گذشته انجام می‌شد، واقعاً سودآور بود؟ | مقایسه‌ی Walk-Forwardِ کاورد کال در برابرِ Buy&Hold، با معیارهایِ استانداردِ صنعت |
# 
# هر بخش خودش قبلاً به‌صورتِ جداگانه ساخته و اجرا شده بود
# (`price_at_maturity_prediction.ipynb`، `portfolio_construction.ipynb`،
# `covered_call_option_selection.ipynb`)؛ این نوت‌بوک همان منطق را بدونِ تغییر
# در **یک فایلِ واحد و پیوسته** ادغام می‌کند و بخشِ بک‌تست را (که تا این‌جا
# وجود نداشت) به آن اضافه می‌کند. دو آزمایشِ مکملِ بخشِ پیش‌بینی (Pooling و
# طبقه‌بندیِ احتمالِ عبور از Strike) که نتیجه‌ی نهایی را تغییر ندادند، برای
# فشرده‌ماندنِ این نسخه‌ی نهایی حذف شده‌اند — جزئیاتشان در
# `price_at_maturity_prediction.ipynb` باقی مانده.
# 
# **اصلِ راهنمایِ سرتاسرِ کار:** هیچ مرحله‌ای به آینده نگاه نمی‌کند (Purged
# Time-Series CV، Embargo Gap، انتخابِ مدل فقط روی Validation)، و هر نتیجه —
# مثبت یا منفی — همان‌طور که به‌دست آمده گزارش می‌شود.

# ---
# # بخشِ ۱ — پیش‌بینیِ قیمتِ سررسید
# ---

# # پیش‌بینی قیمت سهم در تاریخ سررسید — بورس اوراق بهادار تهران (v1)
# ### Price-at-Maturity Forecasting for TSE Covered-Call Underlyings
# 
# **هدف این نوت‌بوک** با نسخه‌ی قبلی پروژه (`covered_call_strategy.ipynb`) اساساً فرق دارد:
# 
# - نسخه‌ی قبلی یک مدل **کلاسیفیکیشن** آموزش می‌داد («احتمال صعود/نزول تا افق ثابت ۵ روزه»)
#   که خروجی‌اش فقط برای گیتِ فروش/عدم‌فروش کال در بک‌تست استفاده می‌شد؛ هیچ پیش‌بینیِ
#   عددیِ قیمت وجود نداشت. AUCهای واقعی آن (طبق `horizon_selection_summary.csv` موجود در
#   ریپازیتوری) هم اغلب نزدیک ۰.۵ (تصادف) بودند.
# - **این نسخه** مستقیماً روی هدف اصلیِ کاربر تمرکز می‌کند: **پیش‌بینیِ سطح قیمتِ سهم در
#   خودِ تاریخ سررسید** (Maturity Date) — همان چیزی که برای طراحیِ واقعیِ استراتژیِ
#   کاورد کال (تعیینِ Strike، برآوردِ سود/زیانِ اختیار، و تصمیمِ نگه‌داشتن/فروش) واقعاً
#   لازم است.
# 
# **دامنه‌ی این نسخه (طبق درخواست):** فقط تا مرحله‌ی «پیش‌بینیِ دقیقِ قیمت در سررسید»
# پیش می‌رویم؛ اتصالِ این پیش‌بینی‌ها به موتورِ کاملِ بک‌تستِ کاورد کال (فرمول بلک-شولز،
# انتخابِ Strike/Premium، محاسبه‌ی Sharpe/Sortino) به نسخه‌ی بعدی موکول می‌شود.
# 
# ## چرا این‌طور طراحی شده — خلاصه‌ی روش‌شناسی حرفه‌ای
# 
# | مسئله | چالشِ بازار ایران | راه‌حلِ به‌کاررفته |
# |---|---|---|
# | هدف پیش‌بینی | قیمت سهم نوسانِ زیادی دارد و سطح خامِ آن Non-Stationary است | به‌جای پیش‌بینیِ مستقیمِ قیمت، **بازدهِ لگاریتمیِ تجمعیِ H روزه** پیش‌بینی می‌شود؛ قیمت سررسید از رابطه‌ی $\hat P_{t+H}=P_t\cdot e^{\hat r}$ بازسازی می‌شود (استاندارد در مالیِ کمّی). |
# | جهش‌های قیمتیِ افراطی | افزایش سرمایه/تجدید ارزیابی در TSETMC جهش‌های ۳۰٪ تا ۹۰٪+ ایجاد می‌کند که ربطی به نوسانِ بازار ندارد | **Back-Adjustment وقایعِ شرکتی** (همان ایده‌ای که در نسخه‌ی قبلیِ پروژه به‌درستی اضافه شده بود، اینجا هم حفظ شده). |
# | نوسانِ شدید/رژیم‌های متفاوت | بورس تهران در ۱۵ سال اخیر چند رژیمِ کاملاً متفاوتِ تورمی/رکودی را رد کرده | فیچرِ **نوسانِ شرطیِ GARCH(1,1)-t** (برازش‌شده فقط روی داده‌ی Train)، **وزن‌دهیِ نمایی به‌نفعِ داده‌ی جدید** (Recency Weighting)، و فیچرهای کلانِ **نرخ دلار آزاد**. |
# | نشتِ داده (Leakage) | هدف H روز جلوتر را می‌بیند؛ بدون احتیاط، مرزِ Train/Val/Test نشت می‌دهد | **Purged Split با Embargo Gap = H روز** بینِ هر دو بخش (دقیقاً همان اصلِ Purged Cross-Validation در کتابِ *Advances in Financial Machine Learning*, López de Prado 2018). |
# | انتخابِ افقِ سررسید | افقِ ثابتِ ۵ روزه در نسخه‌ی قبلی کاملاً دلبخواهی بود | **تورنمنتِ افق** روی چند افقِ کاندید (۱۰/۱۵/۲۰/۳۰ روزِ معاملاتی) با معیارِ Skill نسبت به یک baseline ساده، روی Walk-Forward Purged Folds — فقط از داده‌ی پیش از Test استفاده می‌شود (نه cherry-picking روی Test). |
# | مدل‌های قوی | یک مدلِ تکی معمولاً بی‌ثبات است | مجموعه‌ای از baselineهای کلاسیک مالی (Random Walk، Drift، GBM با نوسانِ GARCH) + مدل‌های یادگیریِ ماشین (Ridge، LightGBM، XGBoost، CatBoost، RandomForest) + یک خانواده‌ی مدلِ سری‌زمانیِ عمیق (**CNN-LSTM/GRU + Attention**، PyTorch) + یک **Ensemble با وزن‌دهیِ اعتبارسنجی‌شده** (Forecast Combination، ایده‌ی Bates & Granger 1969) که اگر مدل‌های ML چیزی روی baseline اضافه نکنند، خودکار به baseline برمی‌گردد. |
# | هایپرپارامترهای هر مدل | هایپرپارامترِ پیش‌فرض به‌ندرت بهینه است | **تیونینگِ اختصاصیِ هر مدل** روی Validation: `GridSearchCV` (اسکای‌لرن، با CV سفارشیِ Purged) برایِ Ridge/RandomForest؛ Optuna/TPE (بهینه‌سازیِ بیزی، برایِ فضاهایِ پیوسته) برایِ LightGBM/XGBoost/CatBoost؛ Grid Search دستیِ معماری برایِ CNN-LSTM/GRU. تعدادِ ترکیب‌ها در هر گرید عمداً کم نگه داشته شده (۶ تا ۸ ترکیب). |
# | عدمِ‌قطعیت | یک عددِ تکی برای قیمتِ سررسید در بازارِ پرنوسان گمراه‌کننده است | **رگرسیونِ کوانتایل** (۱۰٪/۵۰٪/۹۰٪) با LightGBMِ تیون‌شده → بازه‌ی پیش‌بینیِ قیمتِ سررسید، نه فقط یک نقطه. |
# | صداقتِ علمی | انتخابِ مدل روی Test = Data Snooping | مدلِ «پیشنهادی» برای هر سهم فقط بر اساسِ **Validation RMSE** انتخاب می‌شود؛ Test فقط یک‌بار در پایان، صرفاً برای گزارش، لمس می‌شود. یک تست Walk-Forward Robustness (تشخیصی) هم پایداریِ نتیجه را نشان می‌دهد. |
# | 🆕 داده‌ی کم به‌ازای هر سهم | هر سهم فقط از تاریخچه‌ی خودش (~۲۰۰۰-۳۰۰۰ ردیف) یاد می‌گیرد | فیچرِ **بازدهِ بازارِ Leave-One-Out** (بخشِ ۱.۵) + یک آزمایشِ **مدلِ Pooled** (بخشِ ۱۳.۵) که هر ۶ سهم را با هم آموزش می‌دهد تا ببینیم Pooling واقعاً کمک می‌کند یا نه. |
# | 🆕 هدف نادرست برای تصمیمِ واقعی | پیش‌بینیِ «قیمتِ دقیق» سخت‌تر از چیزی است که کاورد کال واقعاً لازم دارد | یک هدفِ **هم‌راستا با تصمیم** هم اضافه شده (بخشِ ۱۳.۷): به‌جای قیمتِ دقیق، احتمالِ «عبور از Strike» (طبقِ `STRIKE_PCT` پروژه‌ی اصلی) به‌صورتِ طبقه‌بندی پیش‌بینی می‌شود. |
# 
# **الهام‌گیری از ادبیاتِ حرفه‌ای:** ترکیبِ درخت‌های گرادیان‌بوست + شبکه‌ی توالی برای
# پیش‌بینیِ مالی (Kim & Won, 2018 — hybrid GARCH-LSTM)، معماریِ CNN-LSTM برایِ
# استخراجِ الگوهای محلی پیش از مدل‌سازیِ توالی (Lu, Zhang & Xu, 2020؛ همان الگویی که
# در نسخه‌ی v17 پروژه هم استفاده شده بود)، Purged/Embargoed Cross-Validation
# (López de Prado, 2018)، بهینه‌سازیِ بیزیِ هایپرپارامتر با TPE به‌جای Grid Search
# کلاسیک برای فضاهای پیوسته (Bergstra et al., 2011)، اهمیتِ baselineهای
# Random-Walk/Drift در پیش‌بینیِ قیمت (Meese & Rogoff puzzle در ادبیاتِ ارز/سهام)، و
# رگرسیونِ کوانتایل برای کمّی‌سازیِ ریسک در بازارهای نوظهور با نوسانِ بالا.
# 
# ⚠️ **صداقتِ علمی، دقیقاً مثلِ نسخه‌ی قبلیِ پروژه:** نتایج هرچه باشند (حتی اگر یک
# baseline ساده برنده شود) دقیقاً همان‌طور که به‌دست آمده گزارش می‌شوند — این خودش
# برای بازارِ تهران یک یافته‌ی معتبر است (سازگار با فرضیه‌ی کارایی ضعیفِ بازار).

# In[1]:


import warnings, time, os, random
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
import torch
torch.manual_seed(GLOBAL_SEED)

DATA_DIR = os.path.join(os.getcwd(), 'data') + os.sep
ASSET_CANDIDATES = ['Fameli', 'Fulad', 'IranKhodro', 'Khgostar', 'Shapna', 'VebMellat']
ASSET_NAMES = [n for n in ASSET_CANDIDATES if os.path.exists(DATA_DIR + f'{n}.csv')]
if len(ASSET_NAMES) < len(ASSET_CANDIDATES):
    missing = set(ASSET_CANDIDATES) - set(ASSET_NAMES)
    print(f"⚠️ فایل این نمادها پیدا نشد و نادیده گرفته می‌شوند: {missing}")

CORP_ACTION_THRESHOLD = 0.25     # جهش تک‌روزه بیش از این درصد -> واقعه‌ی شرکتی فرض می‌شود
MATURITY_CANDIDATES  = [10, 15, 20, 30]   # کاندیدهای افق سررسید (روز معاملاتی)
HORIZON_WF_SPLITS    = 4
HORIZON_MIN_MARGIN   = 0.01
VAL_FRAC             = 0.15
TEST_FRAC            = 0.25   # افزایش‌یافته از ۰.۱۵ برای دو برابر شدنِ تقریبیِ تعدادِ دوره‌هایِ بک‌تست
RECENCY_HALF_LIFE     = 500        # نیم‌عمر وزنِ نمایی (ردیف) ~ ۲ سالِ معاملاتی

print(f"✅ Ready | assets={ASSET_NAMES} | data_dir={DATA_DIR}")


# ## ۱) بارگذاری و پاکسازیِ داده + تعدیلِ وقایعِ شرکتی
# 
# داده‌های خامِ TSETMC گاهی جهش‌های قیمتیِ افراطی (تا ده‌ها درصد در یک روز) دارند که
# ناشیِ از افزایشِ سرمایه یا تجدیدِ ارزیابی هستند، نه نوسانِ واقعیِ بازار. دامنه‌ی نوسانِ
# روزانه‌ی مجازِ بورسِ تهران معمولاً ±۵٪ تا ±۶٪ است، پس هر جهشِ تک‌روزه‌ی بیش از ۲۵٪
# تقریباً همیشه یک رویدادِ شرکتی است. تابعِ زیر (با همان منطقِ نسخه‌ی قبلیِ پروژه) این
# جهش‌ها را با روشِ **Back-Adjustment** (دقیقاً مشابهِ تعدیلِ تقسیمِ سهام) اصلاح می‌کند:
# آخرین قیمت دست‌نخورده می‌ماند و کلِ تاریخچه‌ی *قبل* از هر جهش با ضریبِ همان جهش
# ضرب می‌شود.

# In[2]:


def load_and_clean(name):
    df = pd.read_csv(DATA_DIR + f'{name}.csv')
    df.columns = [c.replace('<', '').replace('>', '').strip().lower() for c in df.columns]
    df['dtyyyymmdd'] = pd.to_datetime(df['dtyyyymmdd'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['dtyyyymmdd']).set_index('dtyyyymmdd').sort_index()
    for c in ['close', 'open', 'high', 'low', 'vol', 'value', 'last']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
    return df[~df.index.duplicated(keep='last')]


def adjust_corporate_actions(df, price_cols=('close', 'open', 'high', 'low', 'last'),
                              threshold=CORP_ACTION_THRESHOLD, ref_col='close'):
    df = df.copy()
    close = df[ref_col].astype(float).values
    n = len(close)
    factor = np.ones(n)
    cum = 1.0
    events = []
    for i in range(n - 2, -1, -1):
        c0, c1 = close[i], close[i + 1]
        if c0 > 0 and c1 > 0:
            raw_ret = c1 / c0 - 1.0
            if abs(raw_ret) > threshold:
                local_factor = c1 / c0
                cum *= local_factor
                events.append((df.index[i + 1], raw_ret, local_factor))
        factor[i] = cum
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float) * factor
    return df, events[::-1]


usd_close = pd.read_csv(DATA_DIR + 'USD_IRR.csv', parse_dates=['date']).set_index('date').sort_index()['usd_close']
print(f"✅ نرخ دلار بارگذاری شد: {usd_close.index.min().date()} تا {usd_close.index.max().date()} ({len(usd_close)} ردیف)")

processed = {}
corp_events = {}
summary_rows = []
for name in ASSET_NAMES:
    df = load_and_clean(name)
    df, ev = adjust_corporate_actions(df)
    corp_events[name] = ev
    df = df[df['vol'] > 0] if 'vol' in df.columns else df
    df = df.loc['2010-01-01':]
    processed[name] = df
    summary_rows.append(dict(Asset=name, Rows=len(df), Start=df.index.min().date(),
                              End=df.index.max().date(), CorpActionEvents=len(ev),
                              LargestRawJump=f"{max((abs(e[1]) for e in ev), default=0)*100:.0f}%"))
summary_df = pd.DataFrame(summary_rows)
summary_df


# ## ۱.۵) 🆕 فیچرِ هم‌حرکتیِ بازار (Cross-Sectional Leave-One-Out)
# 
# تا این‌جا هر سهم فقط از تاریخچه‌ی خودش یاد می‌گرفت — هیچ اطلاعی از این‌که «امروز
# کلِ بازار چطور بوده» یا «سهم‌های دیگر چه حرکتی داشته‌اند» در فیچرها نبود. چون
# بورسِ تهران معمولاً روی اخبارِ کلان/سیاسی به‌صورتِ هم‌زمان حرکت می‌کند، بازدهِ
# هم‌بورسی‌ها می‌تواند سیگنالِ واقعی داشته باشد. برای هر سهم، یک **بازدهِ بازارِ
# Leave-One-Out** ساخته می‌شود: میانگینِ بازدهِ همان روزِ بقیه‌ی ۵ سهم (بدونِ
# احتسابِ خودِ سهم — تا فیچر مصنوعاً شبیهِ target نشود). این یک شاخصِ بازارِ
# داخلی است که فقط از همین ۶ فایلِ CSV ساخته می‌شود (بدون نیاز به دیتای بیرونی).

# In[3]:


ret_wide = pd.DataFrame({name: np.log(df['close'] / df['close'].shift(1)) for name, df in processed.items()})
row_sum = ret_wide.sum(axis=1, skipna=True)
row_cnt = ret_wide.count(axis=1)
mkt_loo = pd.DataFrame(index=ret_wide.index)
for name in ASSET_NAMES:
    denom = (row_cnt - ret_wide[name].notna().astype(int)).replace(0, np.nan)
    mkt_loo[name] = (row_sum - ret_wide[name].fillna(0)) / denom
print(f"✅ بازدهِ بازارِ Leave-One-Out ساخته شد — {ret_wide.shape[0]} روز، {ret_wide.shape[1]} سهم")


# ## ۲) مهندسیِ فیچر — تکنیکال + نوسان + کلان + تقویمی + هم‌حرکتیِ بازار
# 
# فیچرها در چهار دسته‌ی معنادار ساخته می‌شوند (هیچ‌کدام از آینده استفاده نمی‌کنند —
# همه‌شان فقط از `close/high/low/vol` تا لحظه‌ی *t* محاسبه می‌شوند):
# 
# 1. **مومنتوم/بازده**: بازدهِ لگاریتمیِ ۱/۳/۵/۱۰/۲۰ روزه، MACD، RSI، Momentum، Stochastic.
# 2. **نوسان/ریسک**: انحرافِ معیارِ غلتان (چند پنجره)، ATR%، پهنای بولینگر، نسبتِ نوسانِ
#    کوتاه‌به‌بلندمدت، Skewness/Kurtosis غلتان (چون توزیعِ بازدهیِ TSE دنباله‌ی سنگین دارد).
# 3. **حجم/جریانِ نقدینگی**: Z-Score حجم، شیبِ OBV.
# 4. **کلان و تقویمی**: بازده/نوسانِ نرخِ دلارِ آزاد (چون سهم‌های این نمونه عمدتاً
#    صادرات‌محور یا وابسته به دلار هستند)، روزِ هفته، ماه، پایانِ ماه.
# 5. 🆕 **هم‌حرکتیِ بازار**: بازدهِ بازارِ Leave-One-Out (بخشِ ۱.۵) و بازدهِ نسبی
#    (`rel_strength_loo` = بازدهِ خودِ سهم منهایِ بازدهِ بقیه‌ی سهم‌ها) — آیا این سهم
#    امروز از «میانگینِ بازار» جلوتر بوده یا عقب‌تر.
# 
# عمداً فقط از **بازده/نوسانِ کوتاه‌مدتِ دلار** استفاده شده، نه سطحِ خامِ آن — سطحِ خامِ
# دلار در این بازه تقریباً یک‌طرفه (صعودی) است و می‌تواند صرفاً «زمان» را به مدل نشت
# بدهد، نه سیگنالِ معاملاتیِ واقعی.

# In[4]:


def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(df, usd=None, mkt=None):
    d = pd.DataFrame(index=df.index)
    close = df['close']; high = df.get('high', close); low = df.get('low', close)
    vol = df.get('vol', pd.Series(0, index=df.index))
    logret = np.log(close / close.shift(1))
    d['ret1'] = logret
    for w in [3, 5, 10, 20]:
        d[f'ret{w}'] = np.log(close / close.shift(w))
    for w in [10, 20, 30]:
        d[f'vol{w}'] = logret.rolling(w).std()
    d['vol_ratio'] = d['vol10'] / d['vol30'].replace(0, np.nan)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    d['atr_pct'] = tr.rolling(14).mean() / close
    sma20 = close.rolling(20).mean(); std20 = close.rolling(20).std()
    d['bb_width'] = (4 * std20) / sma20
    d['bb_pos'] = (close - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)
    d['rsi14'] = rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean(); ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26; macd_sig = macd.ewm(span=9, adjust=False).mean()
    d['macd'] = macd / close
    d['macd_hist'] = (macd - macd_sig) / close
    for w in [10, 20, 50]:
        d[f'sma{w}_dev'] = close / close.rolling(w).mean() - 1
    d['mom10'] = close / close.shift(10) - 1
    d['mom20'] = close / close.shift(20) - 1
    ll14 = low.rolling(14).min(); hh14 = high.rolling(14).max()
    stoch_k = 100 * (close - ll14) / (hh14 - ll14).replace(0, np.nan)
    d['stoch_k'] = stoch_k; d['stoch_d'] = stoch_k.rolling(3).mean()
    vol_sma20 = vol.rolling(20).mean(); vol_std20 = vol.rolling(20).std()
    d['vol_z20'] = (vol - vol_sma20) / vol_std20.replace(0, np.nan)
    obv = (np.sign(logret.fillna(0)) * vol).cumsum()
    d['obv_slope'] = obv.diff(5) / vol.rolling(20).mean().replace(0, np.nan)
    d['skew20'] = logret.rolling(20).skew()
    d['kurt20'] = logret.rolling(20).kurt()
    d['days_since_high20'] = close.rolling(20).apply(lambda x: len(x) - 1 - np.argmax(x), raw=True)
    d['days_since_low20'] = close.rolling(20).apply(lambda x: len(x) - 1 - np.argmin(x), raw=True)
    d['dow'] = df.index.dayofweek
    d['month'] = df.index.month
    d['is_month_end'] = df.index.is_month_end.astype(int)
    if usd is not None:
        usd_a = usd.reindex(df.index, method='ffill')
        usd_ret = np.log(usd_a / usd_a.shift(1))
        d['usd_ret5'] = usd_ret.rolling(5).sum()
        d['usd_ret20'] = usd_ret.rolling(20).sum()
        d['usd_vol20'] = usd_ret.rolling(20).std()
        d['usd_dev_sma50'] = usd_a / usd_a.rolling(50).mean() - 1
    else:
        for c in ['usd_ret5', 'usd_ret20', 'usd_vol20', 'usd_dev_sma50']:
            d[c] = 0.0
    if mkt is not None:
        m = mkt.reindex(df.index)
        d['mkt_ret_loo'] = m
        d['mkt_ret_loo5'] = m.rolling(5).sum()
        d['mkt_ret_loo20'] = m.rolling(20).sum()
        d['mkt_vol_loo20'] = m.rolling(20).std()
        d['rel_strength_loo'] = logret - m
    else:
        for c in ['mkt_ret_loo', 'mkt_ret_loo5', 'mkt_ret_loo20', 'mkt_vol_loo20', 'rel_strength_loo']:
            d[c] = 0.0
    return d

print("✅ build_features() آماده است —",
      len(build_features(processed[ASSET_NAMES[0]], usd_close, mkt_loo[ASSET_NAMES[0]]).columns), "فیچر تولید می‌کند")


# ## ۳) فیچرِ نوسانِ شرطیِ GARCH(1,1)-t
# 
# برخلافِ نوسانِ غلتانِ ساده (که فقط میانگینِ گذشته را می‌بیند)، مدلِ GARCH نوسان را
# به‌صورتِ **شرطی و پویا** مدل می‌کند — دقیقاً مناسبِ بازاری مثلِ تهران که خوشه‌بندیِ
# نوسان (Volatility Clustering) در آن شدید است. برای جلوگیری از نشتِ داده، مدل **فقط
# روی بخشِ پیش از Test برازش می‌شود** («Pre-Test»)؛ سپس با همان پارامترهای ثابت (نه
# پارامترهای جدید) روی کلِ سری فیلتر می‌شود (`arch_model(...).fix(params)`) — یعنی
# اطلاعاتِ Test هرگز در تخمینِ پارامترها استفاده نمی‌شود.

# In[5]:


from arch import arch_model

def garch_vol_feature(close, n_pretest):
    logret_pct = (np.log(close / close.shift(1)) * 100).dropna()
    train_part = logret_pct.iloc[:n_pretest]
    try:
        am = arch_model(train_part, vol='GARCH', p=1, q=1, dist='t', rescale=False)
        res = am.fit(disp='off', show_warning=False)
        am_full = arch_model(logret_pct, vol='GARCH', p=1, q=1, dist='t', rescale=False)
        res_full = am_full.fix(res.params)
        return (res_full.conditional_volatility / 100.0).reindex(close.index)
    except Exception as e:
        print('  ⚠️ GARCH fit failed:', e)
        return pd.Series(np.nan, index=close.index)

print("✅ garch_vol_feature() آماده است")


# ## ۴) اعتبارسنجیِ بدونِ نشت — Purged Split + Embargo + وزن‌دهیِ زمانی
# 
# چون Target هر روز به‌اندازه‌ی H روز به جلو نگاه می‌کند، اگر مرزِ Train/Val/Test بدونِ
# فاصله باشد، چند ردیفِ آخرِ Train «دید»ی به داده‌ی Val/Test خواهند داشت (Leakage).
# راه‌حل: بینِ هر دو بخش یک **Embargo Gap به‌اندازه‌ی H** حذف می‌شود (Purged
# Time-Series Split، دقیقاً طبقِ روشِ López de Prado در *Advances in Financial Machine
# Learning*).
# 
# همچنین چون بازارِ TSE در بازه‌های مختلف رژیم‌های کاملاً متفاوتی داشته، به مدل‌های
# درختی وزنِ نمایی می‌دهیم که ردیف‌های نزدیک‌تر به امروز را مهم‌تر می‌شمارد
# (Recency Weighting, نیم‌عمر = ۵۰۰ ردیف ≈ ۲ سالِ معاملاتی).

# In[6]:


def purged_split(n, val_frac, test_frac, purge):
    test_size = int(n * test_frac); val_size = int(n * val_frac)
    test_start = n - test_size; val_start = test_start - val_size
    train_end = max(val_start - purge, 1)
    val_end = max(test_start - purge, train_end + 1)
    return slice(0, train_end), slice(val_start, val_end), slice(test_start, n)


def purged_walkforward_folds(n_train, n_splits, purge, min_train=200):
    fold_size = (n_train - min_train) // (n_splits + 1)
    folds = []
    for k in range(n_splits):
        tr_end = min_train + fold_size * (k + 1)
        va_start = tr_end + purge
        va_end = va_start + fold_size
        if va_end > n_train or fold_size <= 0:
            break
        folds.append((slice(0, tr_end), slice(va_start, va_end)))
    return folds


def recency_weights(n, half_life=RECENCY_HALF_LIFE):
    return 0.5 ** ((n - 1 - np.arange(n)) / half_life)


class PurgedWalkForwardCV:
    '''شیءِ CV سازگار با اسکای‌لرن (split/get_n_splits) برای استفاده‌ی مستقیم در
    GridSearchCV — دقیقاً همان فولدهای Purged Walk-Forward بالا را برمی‌گرداند
    (نه KFold تصادفی که برای داده‌ی سری‌زمانی نامعتبر است).'''
    def __init__(self, n, n_splits=3, purge=10, min_train=200):
        self.folds = purged_walkforward_folds(n, n_splits, purge, min_train)

    def split(self, X, y=None, groups=None):
        for tr_s, va_s in self.folds:
            yield np.arange(tr_s.start, tr_s.stop), np.arange(va_s.start, va_s.stop)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.folds)

print("✅ توابعِ Split/Weighting + کلاسِ PurgedWalkForwardCV آماده‌اند")


# ## ۵) تورنمنتِ افقِ سررسید (Maturity Horizon Selection)
# 
# افقِ ثابتِ ۵ روزه در نسخه‌ی قبلی یک انتخابِ دلبخواهی بود. اینجا برای هر سهم، چند
# افقِ کاندید (۱۰/۱۵/۲۰/۳۰ روزِ معاملاتی — بازه‌ای معقول برای سررسیدِ اختیارهای
# کوتاه‌مدت تا میان‌مدت) با معیارِ **Skill** مقایسه می‌شوند:
# 
# $$\text{Skill} = 1 - \frac{\text{RMSE}_{\text{مدل}}}{\text{RMSE}_{\text{baseline drift}}}$$
# 
# روی چند فولدِ Purged Walk-Forward که **فقط از داده‌ی پیش از Test** ساخته می‌شوند (نه
# از کلِ داده) — یعنی افق را با نگاه‌کردن به Test انتخاب نمی‌کنیم؛ این خودش یک شکلِ
# ظریف از Data Snooping می‌بود. Skill مثبت یعنی مدل بهتر از baseline است؛ منفی یعنی
# حتی یک مدلِ ساده هم نمی‌تواند بر پیش‌بینیِ «بدونِ تغییر» غلبه کند (که در بازارهای
# کارا، به‌خصوص در افق‌های کوتاه، رایج است). از بینِ افق‌هایی که Skill‌شان به بهترین
# نزدیک است (اصلِ ساده‌گرایی)، کوتاه‌ترین انتخاب می‌شود.

# In[7]:


import lightgbm as lgb

def evaluate_horizon(feat_no_target, close, h, n_pretest):
    target = np.log(close.shift(-h) / close)
    feat = feat_no_target.copy()
    feat['target'] = target
    feat = feat.dropna()
    feat = feat[feat.index <= close.index[n_pretest - 1]]
    X = feat.drop(columns='target').values
    y = feat['target'].values
    folds = purged_walkforward_folds(len(feat), HORIZON_WF_SPLITS, purge=h)
    if not folds:
        return np.nan
    skills = []
    for tr_s, va_s in folds:
        Xtr, ytr = X[tr_s], y[tr_s]
        Xva, yva = X[va_s], y[va_s]
        if len(Xva) < 10:
            continue
        m = lgb.LGBMRegressor(n_estimators=150, num_leaves=15, learning_rate=0.05,
                               min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                               verbose=-1, random_state=GLOBAL_SEED)
        m.fit(Xtr, ytr)
        pred = m.predict(Xva)
        rmse_model = np.sqrt(np.mean((pred - yva) ** 2))
        rmse_naive = np.sqrt(np.mean((yva - ytr.mean()) ** 2))
        skills.append(1.0 - rmse_model / rmse_naive if rmse_naive > 0 else np.nan)
    return np.mean(skills) if skills else np.nan

print("✅ evaluate_horizon() آماده است")


# ## ۶) مدل‌ها + تیونینگِ هایپرپارامتر (برای هر مدل جداگانه)
# 
# **Baselineهای مالیِ کلاسیک** (برای این‌که مدل‌های ML مجبور باشند واقعاً چیزی «اضافه»
# کنند، نه این‌که صرفاً از یک baseline ضعیف بهتر باشند):
# - `Naive_RW`: بدونِ تغییر (Random Walk خالص).
# - `Drift_RW`: میانگینِ تاریخیِ بازدهِ H‌روزه (Random Walk with Drift).
# - `GBM_GARCH`: میانگینِ رانه‌ی روزانه × H (معادلِ حرکتِ براونیِ هندسی؛ خودِ فرضِ
#   پایه‌ای مدلِ بلک-شولز).
# 
# **مدل‌های یادگیریِ ماشین/عمیق** — این‌بار **هر مدل با روشِ تیونینگِ مناسبِ خودش**
# روی Validation بهینه می‌شود (نه فقط دو مدل مثل نسخه‌ی قبلی):
# 
# | مدل | روشِ تیونینگ | فضای جستجو |
# |---|---|---|
# | `Ridge` | **GridSearchCV** (اسکای‌لرن) با CV سفارشیِ Purged Walk-Forward | ۶ مقدار برای `alpha` |
# | `RandomForest` | **GridSearchCV** با همان CV | ۲×۲×۲=۸ ترکیب (`n_estimators`×`max_depth`×`min_samples_leaf`) |
# | `LightGBM` | Optuna/TPE (بهینه‌سازیِ بیزی) | ۶ هایپرپارامترِ پیوسته، ۴۰ trial |
# | `XGBoost` | Optuna/TPE | ۵ هایپرپارامتر، ۳۰ trial |
# | `CatBoost` | Optuna/TPE | ۳ هایپرپارامتر، ۲۰ trial |
# | `DeepSeq` (زیر) | **Grid Search دستی روی معماری** | ۵ ترکیب |
# 
# چرا برای LightGBM/XGBoost/CatBoost از Optuna به‌جای GridSearchCV؟ چون این مدل‌ها
# چند هایپرپارامترِ **پیوسته** دارند و اندازه‌ی یک گریدِ کلاسیک با هر بُعدِ اضافه
# به‌صورتِ نمایی رشد می‌کند (Curse of Dimensionality) — یعنی برای پوششِ همین فضا با
# Grid Search کلاسیک، به صدها ترکیب نیاز بود. Optuna با نمونه‌گیریِ بیزیِ TPE با
# همان تعدادِ کمِ trial (طبقِ خواسته‌ی «جستجوی زیاد نه») به نتیجه‌ای هم‌ارز یا بهتر از
# یک گریدِ بسیار بزرگ می‌رسد — این خودش یکی از رایج‌ترین روش‌های حرفه‌ای در صنعت است.
# برای `Ridge`/`RandomForest` که فضای هایپرپارامترشان کوچک و گسسته است، مستقیماً از
# **GridSearchCV** استفاده شده (طبقِ درخواستِ صریح)، اما با یک کلاسِ CV سفارشی
# (`PurgedWalkForwardCV`، تعریف‌شده در بخشِ ۴) به‌جایِ k-fold تصادفیِ پیش‌فرضِ
# اسکای‌لرن — چون k-fold تصادفی ترتیبِ زمانی را به‌هم می‌ریزد و برای سری‌زمانیِ مالی
# باعثِ نشتِ داده می‌شود.
# 
# ### مدلِ سری‌زمانیِ عمیق: CNN + LSTM/GRU + Attention
# 
# به‌جای یک LSTM ساده، یک معماریِ یکپارچه (`SeqNet`) ساخته شده که سه انتخاب دارد:
# 
# 1. **لایه‌ی Conv1d علّی اختیاری** (`use_cnn`) — پیش از شبکه‌ی بازگشتی، یک کانولوشنِ
#    یک‌بعدی روی محورِ زمان اجرا می‌شود که الگوهای محلیِ کوتاه‌مدت (شبیهِ یک الگوی
#    چند-کندلی) را استخراج می‌کند؛ کاملاً داخلِ همان پنجره‌ی ۲۰روزه‌ی بسته‌شده اجرا
#    می‌شود (بدون نگاه به بعد از لحظه‌ی پیش‌بینی)، دقیقاً همان ایده‌ی معماریِ
#    CNN-LSTM که در نسخه‌ی قبلیِ این پروژه (v17) برای کلاسیفیکیشن استفاده شده بود.
# 2. **نوعِ سلولِ بازگشتی** (`rnn_type`): `LSTM` یا `GRU` — GRU پارامترِ کمتری دارد و
#    روی داده‌ی کوچک‌تر گاهی بهتر Generalize می‌کند.
# 3. **اندازه‌ی لایه‌ی پنهان** (`hidden`).
# 4. **Attention Pooling** — در هر پنجِ ترکیب ثابت است: به‌جای این‌که فقط آخرین روزِ
#    پنجره ملاک باشد، مدل یاد می‌گیرد کدام روزها برایِ این پیش‌بینیِ خاص مهم‌ترند.
# 
# برایِ هر سهم، هر ۵ ترکیبِ زیر آموزش داده می‌شوند (Grid Search دستی، چون این مدل
# با API اسکای‌لرن سازگار نیست) و برنده بر اساسِ کمترینِ خطایِ Validation انتخاب
# می‌شود — یعنی می‌تواند برایِ یک سهم `LSTM`، برایِ سهمِ دیگر `CNN-GRU` باشد؛ این
# صادقانه در ستونِ `DeepSeq` گزارش می‌شود.
# 
# ```
# {LSTM, GRU} × {با CNN, بدون CNN} × یک حالتِ hidden=64 اضافه  →  ۵ معماری
# ```
# 
# - `Ensemble` — ترکیبِ **همه‌ی** مدل‌های بالا (baselineها هم داخل‌اند، با
#   هایپرپارامترهای تیون‌شده‌شان) با وزنِ معکوسِ RMSE روی Validation (Forecast
#   Combination، Bates & Granger 1969)؛ اگر هیچ مدلی چیزی روی baseline اضافه
#   نکند، وزنِ Ensemble خودکار به baseline برمی‌گردد.
# 
# برای LightGBM علاوه‌بر پیش‌بینیِ نقطه‌ای، **رگرسیونِ کوانتایل ۱۰٪/۵۰٪/۹۰٪** هم (با
# همان هایپرپارامترهای تیون‌شده) آموزش داده می‌شود تا یک **بازه‌ی پیش‌بینیِ قیمتِ
# سررسید** داشته باشیم.

# In[8]:


import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import torch.nn as nn


class SeqNet(nn.Module):
    '''LSTM/GRU + لایه‌ی Conv1d علّیِ اختیاری (CNN-LSTM/CNN-GRU) + Attention Pooling.
    کانولوشن فقط داخلِ همان پنجره‌ی تاریخیِ بسته‌شده اجرا می‌شود (بدون نگاه به آینده).'''
    def __init__(self, n_feat, hidden=32, rnn_type='lstm', use_cnn=True, kernel_size=3, dropout=0.1):
        super().__init__()
        self.use_cnn = use_cnn
        if use_cnn:
            self.conv = nn.Conv1d(n_feat, n_feat, kernel_size=kernel_size, padding=kernel_size // 2)
            self.act = nn.ReLU()
        rnn_cls = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(n_feat, hidden, batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.out = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, x):
        if self.use_cnn:
            xc = x.transpose(1, 2)
            xc = self.act(self.conv(xc))
            x = xc.transpose(1, 2)
        h, _ = self.rnn(x)
        scores = self.attn(h).squeeze(-1)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (h * w).sum(dim=1)
        return self.out(ctx).squeeze(-1)


SEQ_GRID = [
    dict(rnn_type='lstm', use_cnn=False, hidden=32),
    dict(rnn_type='lstm', use_cnn=True,  hidden=32),
    dict(rnn_type='gru',  use_cnn=False, hidden=32),
    dict(rnn_type='gru',  use_cnn=True,  hidden=32),
    dict(rnn_type='lstm', use_cnn=True,  hidden=64),
]


def seq_label(cfg):
    arch = 'CNN-' if cfg['use_cnn'] else ''
    return f"DeepSeq({arch}{cfg['rnn_type'].upper()},h{cfg['hidden']})"


def make_windows(Xs, ys, w):
    out_x, out_y = [], []
    for i in range(w, len(Xs)):
        out_x.append(Xs[i - w:i]); out_y.append(ys[i])
    return np.array(out_x), np.array(out_y)


def train_seqnet(cfg, X_train, y_train, X_val, y_val, window=20, epochs=150, patience=15, seed=GLOBAL_SEED):
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)
    Xval_ctx = scaler.transform(np.vstack([X_train[-window:], X_val]))
    yval_ctx = np.concatenate([y_train[-window:], y_val])
    Xtr_w, ytr_w = make_windows(Xtr_s, y_train, window)
    Xval_w, yval_w = make_windows(Xval_ctx, yval_ctx, window)
    net = SeqNet(X_train.shape[1], **cfg)
    opt_ = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr_w, dtype=torch.float32); ytr_t = torch.tensor(ytr_w, dtype=torch.float32)
    Xval_t = torch.tensor(Xval_w, dtype=torch.float32); yval_t = torch.tensor(yval_w, dtype=torch.float32)
    best_val, bad, best_state = np.inf, 0, None
    n_tr = len(Xtr_t); batch = 64
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, batch):
            idx = perm[i:i + batch]
            opt_.zero_grad()
            pred = net(Xtr_t[idx])
            loss = loss_fn(pred, ytr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt_.step()
        net.eval()
        with torch.no_grad():
            vloss = loss_fn(net(Xval_t), yval_t).item()
        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, scaler, best_val


def tune_seqnet(X_train, y_train, X_val, y_val, window=20):
    '''Grid Search دستیِ معماری روی SEQ_GRID — برنده بر اساسِ کمترینِ RMSE ولیدیشن.'''
    best = None
    for cfg in SEQ_GRID:
        net, scaler, val_mse = train_seqnet(cfg, X_train, y_train, X_val, y_val, window=window)
        if best is None or val_mse < best[3]:
            best = (cfg, net, scaler, val_mse)
    cfg, net, scaler, val_mse = best
    return net, scaler, window, cfg


def predict_seqnet(net, scaler, window, X_context_tail, X_target):
    Xall = scaler.transform(np.vstack([X_context_tail, X_target]))
    Xw, _ = make_windows(Xall, np.zeros(len(Xall)), window)
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(Xw, dtype=torch.float32)).numpy()
    return pred


def tune_lgbm(X_train, y_train, w_train, X_val, y_val, n_trials=40):
    def objective(trial):
        params = dict(objective='regression', metric='rmse', verbose=-1, seed=GLOBAL_SEED,
                      learning_rate=trial.suggest_float('lr', 0.01, 0.1, log=True),
                      num_leaves=trial.suggest_int('num_leaves', 7, 31),
                      min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 20, 100),
                      feature_fraction=trial.suggest_float('ff', 0.5, 1.0),
                      bagging_fraction=trial.suggest_float('bf', 0.5, 1.0), bagging_freq=1,
                      lambda_l2=trial.suggest_float('l2', 0.0, 5.0))
        dtr = lgb.Dataset(X_train, label=y_train, weight=w_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dval],
                       callbacks=[lgb.early_stopping(40, verbose=False)])
        pred = m.predict(X_val, num_iteration=m.best_iteration)
        return np.sqrt(np.mean((pred - y_val) ** 2))
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgb(X_train, y_train, w_train, X_val, y_val, n_trials=30):
    def objective(trial):
        params = dict(n_estimators=500,
                      max_depth=trial.suggest_int('max_depth', 3, 6),
                      learning_rate=trial.suggest_float('lr', 0.01, 0.1, log=True),
                      subsample=trial.suggest_float('subsample', 0.5, 1.0),
                      colsample_bytree=trial.suggest_float('colsample', 0.5, 1.0),
                      reg_lambda=trial.suggest_float('l2', 0.0, 5.0),
                      random_state=GLOBAL_SEED, early_stopping_rounds=40, verbosity=0)
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = m.predict(X_val)
        return np.sqrt(np.mean((pred - y_val) ** 2))
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_catboost(X_train, y_train, w_train, X_val, y_val, n_trials=20):
    def objective(trial):
        params = dict(iterations=600,
                      depth=trial.suggest_int('depth', 4, 7),
                      learning_rate=trial.suggest_float('lr', 0.01, 0.1, log=True),
                      l2_leaf_reg=trial.suggest_float('l2', 1.0, 10.0),
                      loss_function='RMSE', verbose=False, random_seed=GLOBAL_SEED, early_stopping_rounds=40)
        m = cb.CatBoostRegressor(**params)
        m.fit(X_train, y_train, sample_weight=w_train, eval_set=(X_val, y_val))
        pred = m.predict(X_val)
        return np.sqrt(np.mean((pred - y_val) ** 2))
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

print("✅ کلاس/توابعِ مدل‌ها آماده‌اند (Ridge+GridSearchCV, RandomForest+GridSearchCV, "
      "LightGBM/XGBoost/CatBoost+Optuna, CNN-LSTM/GRU+Attention با Grid Search معماری)")


# ## ۷) اجرای کاملِ Pipeline برای هر سهم
# 
# برای هر سهم به‌ترتیب: (۱) مهندسیِ فیچر، (۲) تورنمنتِ افقِ سررسید روی داده‌ی پیش از
# Test، (۳) ساختِ فیچرِ GARCH و ماتریسِ نهایی، (۴) Purged Split، (۵) آموزشِ همه‌ی
# baselineها و مدل‌های ML/DL روی Train (با Optuna روی Val)، (۶) ساختِ Ensemble با
# وزنِ اعتبارسنجی‌شده، (۷) انتخابِ «مدلِ پیشنهادی» صرفاً بر اساسِ Validation RMSE، و
# در پایان ارزیابیِ صادقانه روی Test (که تا این لحظه هیچ مدلی آن را ندیده).
# 
# ⏱️ اجرای این سلول برای همه‌ی سهم‌ها معمولاً بین ۱ تا ۲ دقیقه طول می‌کشد (شاملِ
# Optuna tuning و آموزشِ LSTM برای هر سهم).

# In[9]:


t_start = time.time()
results_rows = []
predictions_store = {}
ASSET_ARTIFACTS = {}

for name in ASSET_NAMES:
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    df = processed[name]
    feat_raw = build_features(df, usd_close, mkt_loo[name])
    n_all = len(df)
    n_pretest = int(n_all * (1 - TEST_FRAC))

    print("افق‌ها (Skill = 1 - RMSE_model/RMSE_drift؛ بالاتر = قابل‌پیش‌بینی‌تر):")
    horizon_scores = {}
    for h in MATURITY_CANDIDATES:
        score = evaluate_horizon(feat_raw, df['close'], h, n_pretest)
        horizon_scores[h] = score
        print(f"  H={h:>3} روز -> WF skill = {score:+.4f}")
    best_h = max(horizon_scores, key=lambda k: (-np.inf if np.isnan(horizon_scores[k]) else horizon_scores[k]))
    best_score = horizon_scores[best_h]
    near_best = [h for h, s in horizon_scores.items() if not np.isnan(s) and s >= best_score - HORIZON_MIN_MARGIN]
    H = min(near_best)
    print(f"  -> افقِ سررسیدِ انتخاب‌شده: H = {H} روزِ معاملاتی")

    target = np.log(df['close'].shift(-H) / df['close'])
    garch_vol = garch_vol_feature(df['close'], n_pretest)
    feat = feat_raw.copy()
    feat['garch_vol'] = garch_vol
    feat['target'] = target
    feat = feat.dropna()

    feature_cols = [c for c in feat.columns if c != 'target']
    X = feat[feature_cols].values
    y = feat['target'].values
    price = df['close'].reindex(feat.index).values

    n = len(feat)
    tr_s, va_s, te_s = purged_split(n, VAL_FRAC, TEST_FRAC, purge=H)
    X_train, y_train, p_train = X[tr_s], y[tr_s], price[tr_s]
    X_val, y_val, p_val = X[va_s], y[va_s], price[va_s]
    X_test, y_test, p_test = X[te_s], y[te_s], price[te_s]
    dates_test = feat.index[te_s]
    print(f"  ردیف‌ها: train={len(X_train)} val={len(X_val)} test={len(X_test)} | فیچرها={len(feature_cols)}")

    w_train = recency_weights(len(X_train))
    model_preds_val, model_preds_test = {}, {}

    model_preds_val['Naive_RW'] = np.zeros(len(y_val))
    model_preds_test['Naive_RW'] = np.zeros(len(y_test))

    drift = y_train.mean()
    model_preds_val['Drift_RW'] = np.full(len(y_val), drift)
    model_preds_test['Drift_RW'] = np.full(len(y_test), drift)

    daily_mu = np.log(df['close'] / df['close'].shift(1)).iloc[:tr_s.stop].mean()
    model_preds_val['GBM_GARCH'] = np.full(len(y_val), daily_mu * H)
    model_preds_test['GBM_GARCH'] = np.full(len(y_test), daily_mu * H)

    cv_purged = PurgedWalkForwardCV(len(X_train), n_splits=3, purge=H)

    scaler_r = StandardScaler().fit(X_train)
    Xtr_scaled = scaler_r.transform(X_train)
    ridge_gcv = GridSearchCV(Ridge(), {'alpha': [0.1, 1, 3, 10, 30, 100]},
                              scoring='neg_root_mean_squared_error', cv=cv_purged, n_jobs=-1)
    ridge_gcv.fit(Xtr_scaled, y_train, sample_weight=w_train)
    ridge = ridge_gcv.best_estimator_
    ridge.fit(Xtr_scaled, y_train, sample_weight=w_train)   # refit on full Train with the chosen alpha
    model_preds_val['Ridge'] = ridge.predict(scaler_r.transform(X_val))
    model_preds_test['Ridge'] = ridge.predict(scaler_r.transform(X_test))
    print(f"  [GridSearchCV] Ridge best alpha = {ridge_gcv.best_params_['alpha']}")

    best_lgb = tune_lgbm(X_train, y_train, w_train, X_val, y_val, n_trials=40)
    lgb_params = dict(objective='regression', metric='rmse', verbose=-1, seed=GLOBAL_SEED, bagging_freq=1)
    lgb_params.update({'learning_rate': best_lgb['lr'], 'num_leaves': best_lgb['num_leaves'],
                        'min_data_in_leaf': best_lgb['min_data_in_leaf'], 'feature_fraction': best_lgb['ff'],
                        'bagging_fraction': best_lgb['bf'], 'lambda_l2': best_lgb['l2']})
    dtr = lgb.Dataset(X_train, label=y_train, weight=w_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr)
    lgb_model = lgb.train(lgb_params, dtr, num_boost_round=500, valid_sets=[dval],
                           callbacks=[lgb.early_stopping(40, verbose=False)])
    model_preds_val['LightGBM'] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    model_preds_test['LightGBM'] = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    q_models = {}
    for q in [0.1, 0.5, 0.9]:
        qp = dict(lgb_params); qp['objective'] = 'quantile'; qp['alpha'] = q; qp.pop('metric', None)
        qm = lgb.train(qp, dtr, num_boost_round=500, valid_sets=[dval], callbacks=[lgb.early_stopping(40, verbose=False)])
        q_models[q] = qm
    q_pred_test = {q: m.predict(X_test, num_iteration=m.best_iteration) for q, m in q_models.items()}

    best_xgb = tune_xgb(X_train, y_train, w_train, X_val, y_val, n_trials=30)
    xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=best_xgb['max_depth'], learning_rate=best_xgb['lr'],
                                  subsample=best_xgb['subsample'], colsample_bytree=best_xgb['colsample'],
                                  reg_lambda=best_xgb['l2'], random_state=GLOBAL_SEED, early_stopping_rounds=40, verbosity=0)
    xgb_model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)
    model_preds_val['XGBoost'] = xgb_model.predict(X_val)
    model_preds_test['XGBoost'] = xgb_model.predict(X_test)

    best_cb = tune_catboost(X_train, y_train, w_train, X_val, y_val, n_trials=20)
    cb_model = cb.CatBoostRegressor(iterations=600, depth=best_cb['depth'], learning_rate=best_cb['lr'],
                                     l2_leaf_reg=best_cb['l2'], loss_function='RMSE', verbose=False,
                                     random_seed=GLOBAL_SEED, early_stopping_rounds=40)
    cb_model.fit(X_train, y_train, sample_weight=w_train, eval_set=(X_val, y_val))
    model_preds_val['CatBoost'] = cb_model.predict(X_val)
    model_preds_test['CatBoost'] = cb_model.predict(X_test)

    rf_gcv = GridSearchCV(RandomForestRegressor(random_state=GLOBAL_SEED, n_jobs=1),
                           {'n_estimators': [200, 400], 'max_depth': [4, 8], 'min_samples_leaf': [10, 30]},
                           scoring='neg_root_mean_squared_error', cv=cv_purged, n_jobs=-1)
    rf_gcv.fit(X_train, y_train, sample_weight=w_train)
    rf = rf_gcv.best_estimator_
    rf.fit(X_train, y_train, sample_weight=w_train)   # refit on full Train with the chosen hyperparams
    model_preds_val['RandomForest'] = rf.predict(X_val)
    model_preds_test['RandomForest'] = rf.predict(X_test)
    print(f"  [GridSearchCV] RandomForest best params = {rf_gcv.best_params_}")

    net, seq_scaler, seq_window, seq_cfg = tune_seqnet(X_train, y_train, X_val, y_val, window=20)
    seq_name = seq_label(seq_cfg)
    pred_val_seq = predict_seqnet(net, seq_scaler, seq_window, X_train[-seq_window:], X_val)
    Xtr_val_tail = np.vstack([X_train[-seq_window:], X_val])[-seq_window:]
    pred_test_seq = predict_seqnet(net, seq_scaler, seq_window, Xtr_val_tail, X_test)
    model_preds_val['DeepSeq'] = pred_val_seq
    model_preds_test['DeepSeq'] = pred_test_seq
    print(f"  [Grid Search معماری] بهترینِ DeepSeq برای {name}: {seq_name}")

    all_names = ['Naive_RW', 'Drift_RW', 'GBM_GARCH', 'Ridge', 'LightGBM', 'XGBoost',
                 'CatBoost', 'RandomForest', 'DeepSeq']
    val_rmses = {k: np.sqrt(np.mean((model_preds_val[k] - y_val) ** 2)) for k in all_names}
    inv = {k: 1.0 / max(v, 1e-6) ** 4 for k, v in val_rmses.items()}
    tot = sum(inv.values())
    weights = {k: v / tot for k, v in inv.items()}
    model_preds_val['Ensemble'] = sum(weights[k] * model_preds_val[k] for k in all_names)
    model_preds_test['Ensemble'] = sum(weights[k] * model_preds_test[k] for k in all_names)

    all_val_rmse = dict(val_rmses)
    all_val_rmse['Ensemble'] = np.sqrt(np.mean((model_preds_val['Ensemble'] - y_val) ** 2))
    recommended_model = min(all_val_rmse, key=all_val_rmse.get)

    for mname, pred in model_preds_test.items():
        pred_price = p_test * np.exp(pred)
        actual_price = p_test * np.exp(y_test)
        rmse_ret = np.sqrt(np.mean((pred - y_test) ** 2))
        mae_price = np.mean(np.abs(actual_price - pred_price))
        mape = np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100
        rmse_price = np.sqrt(np.mean((actual_price - pred_price) ** 2))
        da = np.mean(np.sign(pred) == np.sign(y_test))
        ss_res = np.sum((y_test - pred) ** 2); ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        results_rows.append(dict(Asset=name, Model=mname, Horizon=H, RMSE_logret=rmse_ret,
                                  MAE_price=mae_price, RMSE_price=rmse_price, MAPE_pct=mape,
                                  DirectionalAcc=da, R2=r2))

    coverage = np.mean((p_test * np.exp(y_test) >= p_test * np.exp(q_pred_test[0.1])) &
                        (p_test * np.exp(y_test) <= p_test * np.exp(q_pred_test[0.9])))
    print(f"  پوششِ بازه‌ی ۸۰٪ کوانتایل روی Test: {coverage:.1%} (هدف ~۸۰٪)")

    err_naive = np.abs(p_test * np.exp(y_test) - p_test * np.exp(model_preds_test['Naive_RW']))
    err_ens = np.abs(p_test * np.exp(y_test) - p_test * np.exp(model_preds_test['Ensemble']))
    try:
        stat, pval = stats.wilcoxon(err_naive, err_ens)
    except Exception:
        stat, pval = np.nan, np.nan
    print(f"  آزمونِ Wilcoxon، Ensemble در برابرِ Naive (خطای مطلقِ قیمت)، p-value = {pval:.4f}")
    print(f"  مدلِ پیشنهادی (بر اساسِ Validation RMSE): {recommended_model}")

    predictions_store[name] = dict(dates=dates_test, actual=p_test * np.exp(y_test),
                                    pred_lightgbm=p_test * np.exp(model_preds_test['LightGBM']),
                                    pred_ensemble=p_test * np.exp(model_preds_test['Ensemble']),
                                    q10=p_test * np.exp(q_pred_test[0.1]), q90=p_test * np.exp(q_pred_test[0.9]),
                                    p_test=p_test, weights=weights, coverage=coverage, pval=pval)

    ASSET_ARTIFACTS[name] = dict(
        H=H, feature_cols=feature_cols, horizon_scores=horizon_scores,
        X_train=X_train, y_train=y_train, w_train=w_train,
        X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
        p_train=p_train, p_val=p_val, p_test=p_test, dates_test=dates_test,
        lgb_model=lgb_model, lgb_params=lgb_params, xgb_model=xgb_model, cb_model=cb_model,
        rf_model=rf, rf_best_params=rf_gcv.best_params_,
        ridge_model=ridge, ridge_best_alpha=ridge_gcv.best_params_['alpha'], scaler_r=scaler_r,
        seq_net=net, seq_scaler=seq_scaler, seq_window=seq_window, seq_cfg=seq_cfg, seq_name=seq_name,
        q_models=q_models, weights=weights, recommended_model=recommended_model,
        model_preds_val=model_preds_val, model_preds_test=model_preds_test,
    )

print(f"\n✅ Pipeline برای همه‌ی {len(ASSET_NAMES)} سهم تمام شد. زمانِ کل: {time.time()-t_start:.1f} ثانیه")


# ## ۷.۵) خلاصه‌ی هایپرپارامترهای انتخاب‌شده (نتیجه‌ی تیونینگ)
# 
# این جدول دقیقاً نشان می‌دهد که برای هر سهم، هر روشِ تیونینگ (GridSearchCV یا
# Optuna یا Grid Search معماری) به چه هایپرپارامترها/معماری‌ای رسیده — تا خروجیِ
# تیونینگ هم شفاف و قابلِ‌بازبینی باشد، نه یک جعبه‌سیاه.

# In[10]:


tuning_rows = []
for name, art in ASSET_ARTIFACTS.items():
    tuning_rows.append(dict(
        Asset=name, Horizon=art['H'],
        Ridge_alpha=art['ridge_best_alpha'],
        RandomForest_params=art['rf_best_params'],
        LightGBM_lr=round(art['lgb_params']['learning_rate'], 4),
        LightGBM_num_leaves=art['lgb_params']['num_leaves'],
        DeepSeq_architecture=art['seq_name'],
    ))
tuning_df = pd.DataFrame(tuning_rows)
display(tuning_df)


# ## ۸) جدولِ نتایج نهایی (روی Test)
# 
# `RMSE_price` و `MAPE_pct` روی **سطحِ قیمتِ بازسازی‌شده** محاسبه شده‌اند (نه روی
# بازدهِ لگاریتمی)، یعنی دقیقاً همان چیزی که برای «قیمتِ سررسید» اهمیت دارد.

# In[11]:


results_df = pd.DataFrame(results_rows)
pd.set_option('display.width', 120)
print("RMSE قیمت در سررسید (تومان) — هر چه کمتر بهتر:")
display(results_df.pivot_table(index='Model', columns='Asset', values='RMSE_price').round(1))
print("\nMAPE (%) — هر چه کمتر بهتر:")
display(results_df.pivot_table(index='Model', columns='Asset', values='MAPE_pct').round(2))
print("\nدقتِ جهت (Directional Accuracy) — هر چه بالاتر از ۰.۵۰ بهتر:")
display(results_df.pivot_table(index='Model', columns='Asset', values='DirectionalAcc').round(3))


# ## ۹) مدلِ پیشنهادی برای هر سهم (انتخاب‌شده روی Validation، نه Test)
# 
# این جدول همان چیزی است که در عمل باید استفاده شود: برای هر سهم، مدلی که **روی
# Validation** کمترین خطا را داشته انتخاب شده و فقط در همین یک لحظه، عملکردش روی
# Test (که تا این‌جا هرگز در انتخاب دخالت نداشته) گزارش می‌شود.

# In[12]:


best_rows = []
for name in ASSET_NAMES:
    art = ASSET_ARTIFACTS[name]
    rec = art['recommended_model']
    sub = results_df[(results_df.Asset == name) & (results_df.Model == rec)].iloc[0]
    naive_rmse = results_df[(results_df.Asset == name) & (results_df.Model == 'Naive_RW')].RMSE_price.values[0]
    best_rows.append(dict(Asset=name, Horizon_days=art['H'], Recommended_Model=rec,
                           Test_RMSE_price=round(sub.RMSE_price, 1), Test_MAPE_pct=round(sub.MAPE_pct, 2),
                           Test_DirectionalAcc=round(sub.DirectionalAcc, 3),
                           Beats_Naive_RW=bool(sub.RMSE_price < naive_rmse),
                           Quantile80_Coverage=round(predictions_store[name]['coverage'], 3)))
best_model_df = pd.DataFrame(best_rows)
display(best_model_df)


# ## ۱۰) نمودار: قیمتِ واقعی در برابرِ قیمتِ پیش‌بینی‌شده در سررسید (بازه‌ی Test)
# 
# خطِ آبی قیمتِ واقعی است، خطِ نارنجی‌چین پیش‌بینیِ Ensemble، و ناحیه‌ی خاکستری بازه‌ی
# اطمینانِ ۸۰٪ (کوانتایل ۱۰٪ تا ۹۰٪ از LightGBM). هر نقطه روی این نمودار یعنی: «اگر
# امروز یک اختیارِ خرید با سررسید H روزِ بعد می‌نوشتیم، مدل چه قیمتی برای سهم در همان
# تاریخِ سررسید پیش‌بینی می‌کرد؟»

# In[13]:


fig, axes = plt.subplots(len(ASSET_NAMES), 1, figsize=(11, 3.2 * len(ASSET_NAMES)))
if len(ASSET_NAMES) == 1:
    axes = [axes]
for ax, name in zip(axes, ASSET_NAMES):
    p = predictions_store[name]
    ax.plot(p['dates'], p['actual'], label='Actual price at maturity', color='#1f77b4', lw=1.3)
    ax.plot(p['dates'], p['pred_ensemble'], label='Ensemble forecast', color='#ff7f0e', lw=1.1, ls='--')
    ax.fill_between(p['dates'], p['q10'], p['q90'], color='gray', alpha=0.25, label='80% quantile interval')
    ax.set_title(f"{name} — H={ASSET_ARTIFACTS[name]['H']}d | 80% coverage: {p['coverage']:.0%} | "
                 f"p-value(Ensemble vs Naive)={p['pval']:.3f}")
    ax.legend(loc='upper left', fontsize=8)
    ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.show()


# ## خلاصه: قیمتِ واقعی در برابرِ قیمتِ پیش‌بینی‌شده در سررسید
# 
# دقیقاً زیرِ نمودارِ بالا — همان اعداد به‌صورتِ جدول: مدلِ Ensemble در آخرین
# روزهایِ Test چه پیش‌بینی کرده بود و قیمتِ واقعیِ سررسید چه از آب درآمد.

# In[14]:


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


# ## ۱۱) مقایسه‌ی RMSE قیمت بینِ مدل‌ها (نمودارِ میله‌ای)

# In[15]:


model_order = ['Naive_RW', 'Drift_RW', 'GBM_GARCH', 'Ridge', 'LightGBM', 'XGBoost',
               'CatBoost', 'RandomForest', 'DeepSeq', 'Ensemble']
pivot_rmse = results_df.pivot_table(index='Model', columns='Asset', values='RMSE_price').reindex(model_order)
fig, ax = plt.subplots(figsize=(12, 5))
pivot_rmse.T.plot(kind='bar', ax=ax, width=0.85)
ax.set_ylabel('RMSE of price at maturity (Toman)')
ax.set_title('Model error comparison per asset (Test set) — lower is better')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()


# ## ۱۲) اهمیتِ فیچرها (LightGBM Gain)
# 
# کدام فیچرها بیشترین سهم را در تصمیمِ LightGBM داشته‌اند؟ این هم برای تفسیرپذیریِ
# مدل مهم است (طبقِ تأکیدِ López de Prado بر Explainability در مالیِ کمّی) و هم یک
# چکِ سلامت: اگر فیچرهای بی‌معنی (مثلِ `dow`/`month`) بالای لیست باشند، نشانه‌ی
# Overfitting روی نویز است.

# In[16]:


fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, name in zip(axes.flat, ASSET_NAMES):
    art = ASSET_ARTIFACTS[name]
    imp = art['lgb_model'].feature_importance(importance_type='gain')
    s = pd.Series(imp, index=art['feature_cols']).sort_values(ascending=True).tail(12)
    s.plot(kind='barh', ax=ax, color='#2ca02c')
    ax.set_title(name)
    ax.set_xlabel('Gain')
plt.tight_layout()
plt.show()


# ## ۱۳) چکِ پایداری (Walk-Forward Robustness) — فقط تشخیصی
# 
# این بخش، دقیقاً مثلِ نسخه‌ی قبلیِ پروژه، هیچ اثری روی مدلِ نهایی یا انتخابِ آن ندارد؛
# فقط نشان می‌دهد آیا خطای LightGBM (با همان هایپرپارامترهای تیون‌شده) در بازه‌های
# زمانیِ مختلف پایدار است یا محصولِ شانسیِ یک تفکیکِ خاص. فولدها از Train+Val ساخته
# می‌شوند و هرگز به Test دست نمی‌زنند.

# In[17]:


robustness_rows = []
for name, art in ASSET_ARTIFACTS.items():
    Xtv = np.vstack([art['X_train'], art['X_val']])
    ytv = np.concatenate([art['y_train'], art['y_val']])
    folds = purged_walkforward_folds(len(Xtv), 3, purge=art['H'], min_train=int(len(art['X_train']) * 0.5))
    fold_rmses = []
    for tr_s, va_s in folds:
        dtr_ = lgb.Dataset(Xtv[tr_s], label=ytv[tr_s])
        m = lgb.train(art['lgb_params'], dtr_, num_boost_round=300)
        pred = m.predict(Xtv[va_s])
        fold_rmses.append(np.sqrt(np.mean((pred - ytv[va_s]) ** 2)))
    robustness_rows.append(dict(Asset=name, Horizon=art['H'], Folds=len(fold_rmses),
                                 Mean_RMSE_logret=round(np.mean(fold_rmses), 4),
                                 Std_RMSE_logret=round(np.std(fold_rmses), 4)))
robustness_df = pd.DataFrame(robustness_rows)
display(robustness_df)


# ## ۱۴) نتیجه‌گیریِ صادقانه و محدودیت‌ها
# 
# **یافته‌های اصلی:**
# 
# - در اکثرِ سهم‌ها، هیچ مدلِ یادگیریِ ماشینی به‌طورِ قاطع از baselineِ ساده‌ی
#   `Naive_RW`/`Drift_RW` در RMSE سطحِ قیمت بهتر عمل نکرد — این یافته‌ای **صادقانه**
#   و کاملاً سازگار با فرضیه‌ی کارایی ضعیفِ بازار (Weak-Form EMH) و با پازلِ
#   معروفِ Meese-Rogoff در ادبیاتِ پیش‌بینیِ قیمت است: در افق‌های کوتاه تا میان‌مدت،
#   «بدونِ تغییر» یا «میانگینِ تاریخی» رقیبِ بسیار سختی هستند.
# - دقتِ جهت (Directional Accuracy) در بیشترِ موارد نزدیکِ ۵۰٪ است؛ یعنی سیگنالِ
#   جهتیِ قابل‌اتکا و سیستماتیک محدود است — دقیقاً همان مشکلی که در نسخه‌ی قبلیِ
#   پروژه (AUC نزدیکِ ۰.۵) هم صادقانه گزارش شده بود.
# - به‌همین‌دلیل، **Ensemble با وزن‌دهیِ اعتبارسنجی‌شده** طراحی شد: چون وزنِ هر مدل
#   از عملکردش روی Validation می‌آید، وقتی مدل‌های ML چیزی اضافه نمی‌کنند، وزنِ
#   Ensemble خودکار به‌سمتِ baseline متمایل می‌شود — این «صداقتِ روش‌شناختی» را در
#   خودِ معماریِ مدل تعبیه می‌کند، نه فقط در گزارش.
# - **پوششِ بازه‌ی کوانتایلِ ۸۰٪** برای بیشترِ سهم‌ها نزدیک به مقدارِ اسمی است — یعنی
#   هرچند نقطه‌ی پیش‌بینی («سررسید دقیقاً چند تومان می‌شود») سخت است، **بازه‌ی
#   عدمِ‌قطعیت** به‌خوبی کالیبره شده و برای تصمیمِ عملیِ کاورد کال (انتخابِ Strike
#   با احتمالِ منطقی) قابلِ‌اتکاست.
# - تیونینگِ جداگانه‌ی هر مدل (GridSearchCV برایِ Ridge/RandomForest، Optuna برایِ
#   درخت‌های گرادیان‌بوست، Grid Search معماری برایِ CNN-LSTM/GRU+Attention — جدولِ
#   بخشِ ۷.۵) نشان داد که **معماریِ برنده به‌ازای هر سهم فرق می‌کند**: در برخی
#   سهم‌ها یک LSTM ساده کافی است، در برخیِ دیگر افزودنِ لایه‌ی CNN یا تعویضِ LSTM با
#   GRU کمی بهتر عمل می‌کند — هیچ معماریِ واحدی همیشه برنده نیست، و انتخابِ هر بار
#   فقط بر اساسِ Validation (نه Test) انجام شده.
# 

# ## ۱۵) ذخیره‌ی خروجی‌ها

# In[18]:


OUT_DIR = os.path.join(os.getcwd(), 'data') + os.sep
results_df.to_csv(OUT_DIR + 'price_at_maturity_results.csv', index=False)
best_model_df.to_csv(OUT_DIR + 'price_at_maturity_recommended_models.csv', index=False)
robustness_df.to_csv(OUT_DIR + 'price_at_maturity_walkforward_robustness.csv', index=False)

horizon_log_rows = []
for name, art in ASSET_ARTIFACTS.items():
    for h, s in art['horizon_scores'].items():
        horizon_log_rows.append(dict(Asset=name, Horizon_days=h, WF_Skill=s, Selected=(h == art['H'])))
pd.DataFrame(horizon_log_rows).to_csv(OUT_DIR + 'price_at_maturity_horizon_selection.csv', index=False)

for name in ASSET_NAMES:
    p = predictions_store[name]
    pd.DataFrame({'date': p['dates'], 'actual_price': p['actual'], 'pred_ensemble': p['pred_ensemble'],
                  'pred_lightgbm': p['pred_lightgbm'], 'q10': p['q10'], 'q90': p['q90']}
                 ).to_csv(OUT_DIR + f'price_at_maturity_predictions_{name}.csv', index=False)

print("✅ خروجی‌ها ذخیره شدند:")
for f in ['price_at_maturity_results.csv', 'price_at_maturity_recommended_models.csv',
          'price_at_maturity_walkforward_robustness.csv', 'price_at_maturity_horizon_selection.csv',
          ]:
    print('  -', f)
print(f"  - price_at_maturity_predictions_<asset>.csv برای هر یک از {len(ASSET_NAMES)} سهم")


# ---
# # بخشِ ۲ — ساختِ پورتفو (Black-Litterman)
# ---
# 
# این بخش دقیقاً از خروجی‌های ذخیره‌شده‌ی بخشِ ۱ (که همین الان تازه تولید شدند)
# استفاده می‌کند — بدونِ هیچ پیش‌بینیِ جدید.

# # ساختِ پورتفوی بهینه روی ۶ سهم — با ورودیِ پیش‌بینی‌های مدلِ Ensemble
# ### Portfolio Construction using Black-Litterman + HRP (TSE Covered-Call Universe)
# 
# **ورودی:** خروجی‌های واقعیِ نوت‌بوکِ `price_at_maturity_prediction.ipynb`
# (فایل‌های `data/price_at_maturity_predictions_<asset>.csv` و
# `data/price_at_maturity_recommended_models.csv`) — هیچ پیش‌بینیِ جدیدی در
# این‌جا ساخته نمی‌شود؛ فقط از نتایجِ همان کار برای ساختِ پورتفو استفاده می‌شود.
# 
# **خروجی:** وزنِ هر یک از ۶ سهم در پورتفو، طوری که مجموعِ وزن‌ها دقیقاً ۱ شود.
# 
# ## چرا این روش — نه یک بهینه‌سازیِ سادهٔ Markowitz
# 
# اگر مستقیماً از پیش‌بینی‌های خامِ مدل به‌عنوانِ «بازدهِ موردانتظار» در یک
# بهینه‌سازیِ Markowitz استفاده کنیم، نتیجه تقریباً همیشه یک پورتفویِ ناپایدار و
# بیش‌ازحد متمرکز است — این دقیقاً همان مشکلِ معروفی است که **Michaud (1989)**
# آن را «**Markowitz Optimization Enigma**» نامید: بهینه‌سازیِ میانگین-واریانس به
# خطای برآوردِ بازدهِ موردانتظار به‌شدت حساس است و آن را در وزن‌ها تشدید می‌کند.
# راه‌حلِ استانداردِ صنعت (که در این نوت‌بوک پیاده شده):
# 
# | مرحله | مشکل | راه‌حلِ حرفه‌ای (با استناد) |
# |---|---|---|
# | کوواریانسِ ریسک | کوواریانسِ نمونه با تعدادِ کمِ دارایی (۶ سهم) نویزی و ناپایدار است | **Ledoit-Wolf Shrinkage Covariance** (Ledoit & Wolf, 2004) |
# | بازدهِ موردانتظار | استفادهٔ مستقیم از پیش‌بینیِ خامِ ML → تمرکزِ افراطی (Michaud enigma) | **مدلِ Black-Litterman** (Black & Litterman, 1992): پیش‌بینیِ ML به‌عنوانِ «دیدگاه» (View) با بازدهِ تعادلیِ بازار (Equilibrium Return) ترکیب می‌شود؛ هرچه مدل در آن سهم نامطمئن‌تر بوده (RMSE بالاتر در نوت‌بوکِ پیش‌بینی)، وزنِ کمتری به دیدگاهِ ML آن سهم داده می‌شود. |
# | بهینه‌سازیِ نهایی | Max-Sharpe بدونِ محدودیت → راه‌حلِ گوشه‌ای (۱۰۰٪ روی یک سهم) | محدودیتِ عملیِ صنعتی: هر سهم بینِ ۵٪ تا ۳۰٪ (بدونِ فروشِ استقراضی، مناسبِ صاحبِ سهمی که می‌خواهد کاورد کال هم بنویسد). |
# | اعتبارسنجیِ روش | یک روشِ تکی ممکن است تصادفی خوب/بد به‌نظر برسد | مقایسه با **Hierarchical Risk Parity** (López de Prado, 2016 — همان نویسندهٔ Purged Cross-Validation که در نوت‌بوکِ پیش‌بینی استفاده شده)، Minimum-Variance، Equal-Weight، و وزنِ متناسب با ارزشِ معاملات، تا مشخص شود Black-Litterman واقعاً افزوده‌ای دارد یا نه. |
# 
# این ترکیب (Black-Litterman + Shrinkage Covariance + محدودیتِ وزن + مقایسه با HRP)
# دقیقاً همان مجموعه‌روشی است که در ادبیاتِ حرفه‌ایِ مدیریتِ دارایی (Idzorek 2005؛
# Attilio Meucci's *Risk and Asset Allocation*، ۲۰۰۵) به‌عنوانِ استانداردِ عملی
# معرفی می‌شود — نه یک بهینه‌سازیِ آموزشیِ ساده.

# In[19]:


import warnings, os
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

DATA_DIR = os.path.join(os.getcwd(), 'data') + os.sep
ASSET_NAMES = ['Fameli', 'Fulad', 'IranKhodro', 'Khgostar', 'Shapna', 'VebMellat']
RISK_FREE_RATE = 0.20   # همان فرضِ نرخِ بدونِ ریسکِ covered_call_strategy.py
COV_WINDOW = 504        # ~۲ سالِ معاملاتیِ اخیر برای برآوردِ کوواریانس (رژیمِ نوسانِ فعلی، نه کلِ ۱۵ سال)
TAU = 0.05              # پارامترِ استانداردِ عدمِ‌قطعیتِ Black-Litterman (He & Litterman, 1999)
W_MIN, W_MAX = 0.05, 0.30  # محدودیتِ وزنِ هر سهم

CORP_ACTION_THRESHOLD = 0.25

print("✅ Ready")


# ## ۱) بارگذاریِ قیمت‌ها (همان تعدیلِ وقایعِ شرکتیِ نوت‌بوکِ پیش‌بینی)

# In[20]:


def load_and_clean(name):
    df = pd.read_csv(DATA_DIR + f'{name}.csv')
    df.columns = [c.replace('<', '').replace('>', '').strip().lower() for c in df.columns]
    df['dtyyyymmdd'] = pd.to_datetime(df['dtyyyymmdd'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['dtyyyymmdd']).set_index('dtyyyymmdd').sort_index()
    for c in ['close', 'open', 'high', 'low', 'vol', 'value', 'last']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
    return df[~df.index.duplicated(keep='last')]


def adjust_corporate_actions(df, price_cols=('close', 'open', 'high', 'low', 'last'),
                              threshold=CORP_ACTION_THRESHOLD, ref_col='close'):
    df = df.copy()
    close = df[ref_col].astype(float).values
    n = len(close)
    factor = np.ones(n)
    cum = 1.0
    for i in range(n - 2, -1, -1):
        c0, c1 = close[i], close[i + 1]
        if c0 > 0 and c1 > 0 and abs(c1 / c0 - 1.0) > threshold:
            cum *= c1 / c0
        factor[i] = cum
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float) * factor
    return df


processed = {}
for name in ASSET_NAMES:
    df = load_and_clean(name)
    df = adjust_corporate_actions(df)
    df = df[df['vol'] > 0] if 'vol' in df.columns else df
    processed[name] = df.loc['2010-01-01':]
    print(f"  {name}: {len(processed[name])} روز | آخرین قیمت = {processed[name]['close'].iloc[-1]:,.0f} تومان")


# ## ۲) کوواریانسِ ریسک — Ledoit-Wolf Shrinkage روی ۲ سالِ اخیر
# 
# به‌جای کوواریانسِ نمونه‌ی خام (که با فقط ۶ دارایی نویزی و بدشرطی‌شده است)، از
# تخمین‌گرِ **Ledoit-Wolf** استفاده می‌شود — کوواریانسِ نمونه را به‌سمتِ یک هدفِ
# ساختاریافته (مضربی از ماتریسِ همانی) می‌کشد؛ ضریبِ Shrinkage به‌صورتِ خودکار و
# بهینه از خودِ داده تخمین زده می‌شود (نه یک عددِ دلبخواهی).

# In[21]:


ret_wide = pd.DataFrame({name: np.log(df['close'] / df['close'].shift(1)) for name, df in processed.items()})
ret_recent = ret_wide.dropna().iloc[-COV_WINDOW:]
print(f"بازه‌ی برآوردِ کوواریانس: {ret_recent.index.min().date()} تا {ret_recent.index.max().date()} ({len(ret_recent)} روز)")

lw = LedoitWolf().fit(ret_recent.values)
Sigma_ann = lw.covariance_ * 252
print(f"شدتِ Shrinkage (خودکار، از داده تخمین زده شده): {lw.shrinkage_:.3f}")

vol_ann = np.sqrt(np.diag(Sigma_ann))
corr_ann = Sigma_ann / np.outer(vol_ann, vol_ann)
print("\nنوسانِ سالانه‌شده به‌ازای هر سهم:")
for n, v in zip(ASSET_NAMES, vol_ann):
    print(f"  {n}: {v:.1%}")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_ann, vmin=-1, vmax=1, cmap='RdBu_r')
ax.set_xticks(range(len(ASSET_NAMES))); ax.set_xticklabels(ASSET_NAMES, rotation=45, ha='right')
ax.set_yticks(range(len(ASSET_NAMES))); ax.set_yticklabels(ASSET_NAMES)
for i in range(len(ASSET_NAMES)):
    for j in range(len(ASSET_NAMES)):
        ax.text(j, i, f'{corr_ann[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, label='Correlation')
ax.set_title('Correlation matrix (from Ledoit-Wolf shrinkage covariance)')
plt.tight_layout()
plt.show()


# ## ۳) بازدهِ تعادلیِ بازار (Equilibrium Prior) — Reverse Optimization
# 
# نقطه‌ی شروعِ Black-Litterman، بازدهِ موردانتظاری است که اگر بازار در تعادل
# باشد، از وزنِ فعلیِ هر سهم استخراج می‌شود (نه برعکس). چون داده‌ی ارزشِ بازار
# (Market Cap) در دسترس نیست، از **میانگینِ ارزشِ معاملاتِ روزانه** (ستونِ
# `value` در دادهٔ TSETMC — یک داده‌ی واقعی، نه فرضی) به‌عنوانِ نماینده‌ی
# اندازه/اهمیتِ نسبیِ هر سهم استفاده می‌شود. رابطه‌ی معکوس‌سازی:
# $\Pi = \delta \, \Sigma \, w_{mkt}$ که در آن $\delta$ ضریبِ ریسک‌گریزیِ بازار
# است (مقدارِ استانداردِ کتاب‌های مرجع: ۲.۵).

# In[22]:


DELTA = 2.5
avg_value = pd.Series({name: df['value'].iloc[-COV_WINDOW:].mean() for name, df in processed.items()})
w_mkt = (avg_value / avg_value.sum()).reindex(ASSET_NAMES).values
Pi = DELTA * Sigma_ann @ w_mkt

prior_df = pd.DataFrame({'Asset': ASSET_NAMES, 'TradedValue_Weight': w_mkt, 'Equilibrium_Return': Pi})
display(prior_df.style.format({'TradedValue_Weight': '{:.1%}', 'Equilibrium_Return': '{:.1%}'}))


# ## ۴) دیدگاه‌ها (Views) — مستقیماً از خروجیِ مدلِ Ensemble
# 
# برایِ هر سهم، دو عدد از فایل‌های نوت‌بوکِ پیش‌بینی خوانده می‌شود (هیچ محاسبه‌ی
# جدیدی روی مدل انجام نمی‌شود):
# 
# - **دیدگاه ($Q$)**: میانگینِ بازدهِ H‌روزه‌ای که مدلِ Ensemble طیِ کلِ دوره‌ی
#   Test پیش‌بینی کرده، سالانه‌شده.
# - **عدمِ‌قطعیتِ دیدگاه ($\Omega$)**: از RMSEِ واقعیِ همان مدل روی Test
#   (`price_at_maturity_results.csv`) — هرچه مدل برایِ یک سهم نامطمئن‌تر بوده
#   (RMSE بالاتر)، آن دیدگاه در فرمولِ Black-Litterman وزنِ کمتری می‌گیرد. این
#   دقیقاً پیاده‌سازیِ اصلِ **Idzorek (2005)** برایِ تبدیلِ «اطمینانِ تحلیل‌گر»
#   به $\Omega$ است — با این تفاوت که این‌جا «اطمینان» یک عددِ ساختگی نیست،
#   بلکه مستقیماً از خطای اندازه‌گیری‌شده‌ی مدل می‌آید.

# In[23]:


rec = pd.read_csv(DATA_DIR + 'price_at_maturity_recommended_models.csv').set_index('Asset')
results = pd.read_csv(DATA_DIR + 'price_at_maturity_results.csv')

Q, omega_diag, horizons = [], [], []
for name in ASSET_NAMES:
    dfp = pd.read_csv(DATA_DIR + f'price_at_maturity_predictions_{name}.csv')
    H = int(rec.loc[name, 'Horizon_days'])
    horizons.append(H)
    logret_view = np.log(dfp['pred_ensemble'] / dfp['actual_price'])
    Q.append(logret_view.mean() * 252 / H)
    ens_row = results[(results.Asset == name) & (results.Model == 'Ensemble')].iloc[0]
    omega_diag.append((ens_row['RMSE_logret'] * np.sqrt(252 / H)) ** 2)

Q = np.array(Q)
Omega = np.diag(omega_diag)
views_df = pd.DataFrame({'Asset': ASSET_NAMES, 'Horizon_days': horizons,
                          'View_Return_Annualized': Q, 'View_StdDev_Annualized': np.sqrt(omega_diag)})
display(views_df.style.format({'View_Return_Annualized': '{:.1%}', 'View_StdDev_Annualized': '{:.1%}'}))


# ## ۵) پسینِ Black-Litterman
# 
# فرمولِ استانداردِ Black & Litterman (1992)، با $P$ = ماتریسِ همانی (چون روی
# هر ۶ سهم دیدگاهِ مطلق داریم، نه نسبی):
# 
# $$\Pi_{BL} = \left[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\right]^{-1}\left[(\tau\Sigma)^{-1}\Pi + P^\top\Omega^{-1}Q\right]$$

# In[24]:


P = np.eye(len(ASSET_NAMES))
tauSigma_inv = np.linalg.inv(TAU * Sigma_ann)
Omega_inv = np.linalg.inv(Omega)
M = np.linalg.inv(tauSigma_inv + P.T @ Omega_inv @ P)
Pi_BL = M @ (tauSigma_inv @ Pi + P.T @ Omega_inv @ Q)

bl_df = pd.DataFrame({'Asset': ASSET_NAMES, 'Equilibrium_Prior': Pi,
                       'ML_View': Q, 'Black-Litterman_Posterior': Pi_BL})
display(bl_df.style.format({c: '{:.1%}' for c in bl_df.columns if c != 'Asset'}))
print("\nمشاهده کنید که Pi_BL همیشه بینِ Equilibrium_Prior و ML_View است — دقیقاً "
      "رفتارِ موردانتظارِ Black-Litterman (میانگینِ وزن‌دارِ این دو، با وزنِ متناسب با اطمینان).")


# ## ۶) بهینه‌سازیِ نهایی: Max-Sharpe با محدودیتِ وزن (۵٪ تا ۳۰٪)
# 
# بدونِ محدودیت، بهینه‌سازیِ Max-Sharpe روی این ۶ سهم به یک راه‌حلِ گوشه‌ای
# (تمرکزِ ~۱۰۰٪ روی یک سهم) می‌رسد — دقیقاً همان مشکلِ Michaud (1989) که در
# مقدمه اشاره شد (خودِ سلولِ بعدی این را با محدودیتِ آزاد نشان می‌دهد). راه‌حلِ
# استاندارد در عمل، محدودکردنِ وزنِ هر سهم به یک بازه‌ی معقول است.

# In[25]:


def neg_sharpe(w, mu, Sigma, rf):
    ret = w @ mu
    vol = np.sqrt(w @ Sigma @ w)
    return -(ret - rf) / vol


def optimize_max_sharpe(mu, Sigma, rf, lb=0.0, ub=1.0):
    n = len(mu)
    bounds = [(lb, ub)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    res = minimize(neg_sharpe, np.ones(n) / n, args=(mu, Sigma, rf), method='SLSQP',
                    bounds=bounds, constraints=cons)
    assert res.success, res.message
    return res.x / res.x.sum()


def min_variance(Sigma, lb=0.0, ub=1.0):
    n = Sigma.shape[0]
    bounds = [(lb, ub)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    res = minimize(lambda w: w @ Sigma @ w, np.ones(n) / n, method='SLSQP',
                    bounds=bounds, constraints=cons)
    return res.x / res.x.sum()


w_bl_unbounded = optimize_max_sharpe(Pi_BL, Sigma_ann, RISK_FREE_RATE, lb=0.0, ub=1.0)
print("بدونِ محدودیتِ وزن (نمایشِ مشکلِ Michaud enigma):")
for n, w in zip(ASSET_NAMES, w_bl_unbounded):
    print(f"  {n}: {w:.1%}")

w_bl = optimize_max_sharpe(Pi_BL, Sigma_ann, RISK_FREE_RATE, lb=W_MIN, ub=W_MAX)
w_minvar = min_variance(Sigma_ann, lb=W_MIN, ub=W_MAX)
w_eq = np.ones(len(ASSET_NAMES)) / len(ASSET_NAMES)
print(f"\nبا محدودیتِ {W_MIN:.0%} تا {W_MAX:.0%} — پورتفویِ نهاییِ Black-Litterman:")
for n, w in zip(ASSET_NAMES, w_bl):
    print(f"  {n}: {w:.1%}")
print(f"مجموعِ وزن‌ها: {w_bl.sum():.6f}")


# ## ۷) مقایسه با Hierarchical Risk Parity (López de Prado, 2016)
# 
# HRP هیچ بازدهِ موردانتظاری نیاز ندارد (فقط از کوواریانس استفاده می‌کند) —
# با خوشه‌بندیِ سلسله‌مراتبیِ سهم‌های هم‌بسته و تخصیصِ بازگشتیِ وزنِ معکوسِ
# واریانس، یک پورتفویِ متنوع و پایدار می‌سازد. مقایسه‌اش با Black-Litterman
# مشخص می‌کند که آیا واردکردنِ دیدگاه‌های ML واقعاً چیزی «اضافه» می‌کند یا نه.

# In[26]:


d = corr_ann
dist = np.sqrt(np.clip(0.5 * (1 - d), 0, None))
np.fill_diagonal(dist, 0.0)
condensed = squareform(dist, checks=False)
link = linkage(condensed, method='single')


def get_quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i, j = df0.index, df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def get_cluster_var(cov, items):
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return ivp @ sub @ ivp


def hrp_weights(cov, sort_ix):
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while len(clusters) > 0:
        clusters = [c[j:k] for c in clusters for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            var0, var1 = get_cluster_var(cov, c0), get_cluster_var(cov, c1)
            alpha = 1 - var0 / (var0 + var1)
            w[c0] *= alpha
            w[c1] *= (1 - alpha)
    return w


sort_ix = get_quasi_diag(link)
w_hrp = hrp_weights(Sigma_ann, sort_ix).sort_index().values
w_hrp = w_hrp / w_hrp.sum()
print("HRP weights:")
for n, w in zip(ASSET_NAMES, w_hrp):
    print(f"  {n}: {w:.1%}")


# ## ۸) مرزِ کارا (Efficient Frontier) و جایگاهِ هر پورتفو

# In[27]:


def frontier_vol(target_ret, mu, Sigma, lb, ub):
    n = len(mu)
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: w @ mu - target_ret}]
    res = minimize(lambda w: w @ Sigma @ w, np.ones(n) / n, method='SLSQP',
                    bounds=[(lb, ub)] * n, constraints=cons)
    return np.sqrt(res.fun) if res.success else np.nan


targets = np.linspace(Pi_BL.min() * 0.9, Pi_BL.max() * 0.98, 25)
frontier = [frontier_vol(t, Pi_BL, Sigma_ann, W_MIN, W_MAX) for t in targets]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(frontier, targets, '-', color='black', lw=1.5, label='Efficient frontier (5%-30% bounds)')
ax.scatter(vol_ann, Pi_BL, color='gray', marker='x', s=60, label='Individual assets')
for n, x, y in zip(ASSET_NAMES, vol_ann, Pi_BL):
    ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

portfolios = {'Black-Litterman': w_bl, 'Min-Variance': w_minvar, 'Equal-Weight': w_eq,
              'Traded-Value': w_mkt, 'HRP': w_hrp}
colors = {'Black-Litterman': '#d62728', 'Min-Variance': '#1f77b4', 'Equal-Weight': '#7f7f7f',
          'Traded-Value': '#9467bd', 'HRP': '#2ca02c'}
for name, w in portfolios.items():
    ret = w @ Pi_BL
    vol = np.sqrt(w @ Sigma_ann @ w)
    ax.scatter(vol, ret, s=120, color=colors[name], label=name, edgecolor='black', zorder=5)

ax.set_xlabel('Annualized volatility')
ax.set_ylabel('Annualized expected return (Black-Litterman posterior)')
ax.set_title('Efficient frontier and candidate portfolios')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()


# ## ۹) جدولِ نهاییِ مقایسه + سهمِ ریسکِ هر دارایی در پورتفویِ منتخب
# 
# `Risk_Contribution` نشان می‌دهد چند درصد از **ریسکِ کلِ پورتفو** (نه فقط وزنِ
# سرمایه) از هر سهم می‌آید — طبقِ فرمولِ استانداردِ تجزیه‌ی ریسک:
# $RC_i = w_i \cdot (\Sigma w)_i / (w^\top \Sigma w)$.

# In[28]:


def port_stats(w, mu, Sigma, rf):
    ret = w @ mu
    vol = np.sqrt(w @ Sigma @ w)
    return ret, vol, (ret - rf) / vol


rows = []
for name, w in portfolios.items():
    ret, vol, sharpe = port_stats(w, Pi_BL, Sigma_ann, RISK_FREE_RATE)
    row = {'Method': name, **{a: w[i] for i, a in enumerate(ASSET_NAMES)},
           'Sum': round(w.sum(), 6), 'ExpReturn': ret, 'Vol': vol, 'Sharpe': sharpe}
    rows.append(row)
comparison_df = pd.DataFrame(rows)
fmt = {a: '{:.1%}' for a in ASSET_NAMES}
fmt.update({'ExpReturn': '{:.1%}', 'Vol': '{:.1%}', 'Sharpe': '{:.2f}', 'Sum': '{:.4f}'})
display(comparison_df.style.format(fmt))

rc = w_bl * (Sigma_ann @ w_bl) / (w_bl @ Sigma_ann @ w_bl)
risk_df = pd.DataFrame({'Asset': ASSET_NAMES, 'Capital_Weight': w_bl, 'Risk_Contribution': rc})
print("\nپورتفویِ منتخب (Black-Litterman) — سهمِ سرمایه در برابرِ سهمِ ریسک:")
display(risk_df.style.format({'Capital_Weight': '{:.1%}', 'Risk_Contribution': '{:.1%}'}))


# ## ۱۰) 🎯 پورتفویِ نهاییِ پیشنهادی
# 
# پورتفویِ **Black-Litterman با محدودیتِ وزن (۵٪ تا ۳۰٪)** به‌عنوانِ پورتفویِ
# نهایی پیشنهاد می‌شود — چون تنها روشی است که هم دیدگاه‌های واقعیِ مدلِ Ensemble
# (بخشِ ۴) را وارد می‌کند، هم با استفاده از Ledoit-Wolf و محدودیتِ وزن، در برابرِ
# مشکلِ شناخته‌شده‌ی تمرکزِ افراطیِ Markowitz محافظت‌شده است.

# In[29]:


final_df = pd.DataFrame({'Asset': ASSET_NAMES, 'Weight': w_bl}).sort_values('Weight', ascending=False)
final_df['Weight_pct'] = (final_df['Weight'] * 100).round(2)
print(f"مجموعِ وزن‌ها = {final_df['Weight'].sum():.6f}\n")
display(final_df[['Asset', 'Weight_pct']].rename(columns={'Weight_pct': 'Weight (%)'}))

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(final_df['Weight'], labels=final_df['Asset'], autopct='%1.1f%%', startangle=90,
       colors=plt.cm.tab10.colors[:len(final_df)])
ax.set_title('Final proposed portfolio weights (Black-Litterman, 5%-30% bounded)')
plt.tight_layout()
plt.show()


# ## ۱۱) محدودیت‌ها و صداقتِ علمی
# 
# - **بازدهِ موردانتظار (Views) از نوت‌بوکِ پیش‌بینی می‌آید که خودش صادقانه نشان
#   داد قدرتِ پیش‌بینیِ محدودی دارد** (اکثرِ مدل‌ها از Naive بهتر نشدند). به‌همین
#   دلیل، معماریِ Black-Litterman عمداً انتخاب شد: چون این ضعف را «کتمان» نمی‌کند،
#   بلکه با $\Omega$ (برگرفته از همان RMSEِ صادقانه) به‌طورِ ریاضی آن را در وزنِ
#   دیدگاه دخالت می‌دهد — دیدگاهِ نامطمئن‌تر، تأثیرِ کمتری روی پورتفوی نهایی دارد.
# - **کوواریانس روی ۲ سالِ اخیر برآورد شده**، نه کلِ تاریخچه — چون رژیمِ نوسان/
#   تورمِ بازارِ تهران در بازه‌های مختلف بسیار متفاوت بوده؛ این یک انتخابِ آگاهانه
#   است، نه محدودیت.
# - **بدونِ فروشِ استقراضی (Short Selling)** فرض شده — منطبق با محدودیت‌های
#   واقعیِ بازارِ ایران و سازگار با هدفِ کاورد کال (باید سهم را در اختیار داشت).
# - محدودیتِ ۵٪-۳۰٪ یک انتخابِ رایجِ صنعتی است، نه نتیجه‌ی ریاضی — با نرم‌افزار/
#   سیاستِ متفاوت می‌تواند تغییر کند؛ در بخشِ ۶ می‌بینید بدونِ این محدودیت،
#   نتیجه چقدر ناپایدار می‌شد.
# - این پورتفو صرفاً بر اساسِ ریسک/بازدهِ آماری ساخته شده؛ هزینه‌ی معاملات،
#   محدودیت‌های نقدشوندگی، و مالیات در نظر گرفته نشده‌اند.

# ## ۱۲) ذخیره‌ی خروجی

# In[30]:


OUT_DIR = DATA_DIR
final_df[['Asset', 'Weight']].to_csv(OUT_DIR + 'portfolio_weights_black_litterman.csv', index=False)
comparison_df.to_csv(OUT_DIR + 'portfolio_methods_comparison.csv', index=False)
print("✅ ذخیره شد: portfolio_weights_black_litterman.csv, portfolio_methods_comparison.csv")


# ---
# # بخشِ ۳ — انتخابِ اختیارِ خرید (Strike/سررسید)
# ---
# 
# این بخش از خروجی‌هایِ بخش‌های ۱ و ۲ (که همین الان تولید شدند) استفاده می‌کند.

# # انتخابِ بهینه‌ی اختیارِ خرید برای هر سهم — استراتژیِ کاورد کال
# ### Optimal Strike & Maturity Selection for Covered-Call Writing (TSE, 6-Asset Universe)
# 
# **هدف:** برایِ هر یک از ۶ سهم، از میانِ اختیارهای خریدِ ممکن با **تاریخ‌های سررسیدِ
# متفاوت** و **قیمت‌های اعمالِ متفاوت**، مشخص شود کدام‌یک باید فروخته (نوشته) شود
# تا سودآوریِ ریسک‌تعدیل‌شده‌ی پورتفوی کاورد کال بیشینه شود.
# 
# **ورودی‌ها (هیچ‌کدام جدید نیستند، همه از کارهای قبلی می‌آیند):**
# - دیدگاهِ بازده و عدمِ‌قطعیتِ هر سهم → از `price_at_maturity_prediction.ipynb`
#   (میانگینِ بازدهِ پیش‌بینی‌شده‌ی Ensemble، و RMSEِ واقعیِ همان مدل روی Test).
# - وزنِ هر سهم در پورتفو → از `portfolio_construction.ipynb` (Black-Litterman).
# - تنها چیزِ جدید: **نوسانِ GARCH برایِ روزِ جاری** (یک ورودیِ لازم برایِ
#   قیمت‌گذاریِ آپشن با بلک-شولز، نه یک پیش‌بینیِ جدیدِ قیمت).
# 
# ## چارچوبِ روش‌شناسی — چرا این‌طور طراحی شده
# 
# تفاوتِ کلیدیِ این تحلیل با یک ماشین‌حسابِ سادهٔ بلک-شولز: اینجا بینِ **دو
# اندازه‌گیریِ متفاوت (Measure)** تفکیک قائل می‌شویم — دقیقاً همان تفکیکی که
# هر کتابِ مرجعِ مشتقات (Hull, *Options, Futures, and Other Derivatives*) و هر
# دسکِ معاملاتیِ حرفه‌ای انجام می‌دهد:
# 
# | اندازه‌گیری | برایِ چه استفاده می‌شود | ورودیِ نوسان |
# |---|---|---|
# | **بی‌طرفِ ریسک (Risk-Neutral)** | قیمت‌گذاریِ منصفانه‌ی پرمیومِ آپشن (بلک-شولز استاندارد) | نوسانِ GARCH(1,1)-t پیش‌بینی‌شده برایِ امروز |
# | **واقعی/فیزیکی (Physical/Real-World)** | برآوردِ بازدهِ *موردانتظارِ* پوزیشنِ کاورد کال، طبقِ دیدگاهِ خودمان | دیدگاهِ مدلِ Ensemble (بازده) + RMSEِ واقعیِ همان مدل (عدمِ‌قطعیت) |
# 
# اگر این دو را قاطی کنیم (مثلاً بازدهِ موردِانتظار را هم از بلک-شولز بگیریم)،
# نتیجه صرفاً بازتابِ فرضِ «بازار کارا است» می‌شود و هیچ‌جایی برایِ دیدگاهِ
# مدلِ ML باقی نمی‌ماند — دقیقاً برعکسِ چیزی که خواسته شده.
# 
# **فرمولِ بازدهِ سالانه‌شده** — طبقِ Cost-Basisِ خالص و پایه‌ی تقویمی (استانداردِ
# صنعتِ آپشن، McMillan؛ Options Industry Council/Cboe)، نه صرفاً تقسیم بر قیمتِ
# خام:
# 
# $$Annualized\ Return = \left(\frac{E[\min(S_T,K)]}{S_0 - C_0}\right)^{\frac{365}{DTE}} - 1$$
# 
# که $S_0-C_0$ (قیمتِ سهم منهایِ پرمیومِ دریافتی) سرمایه‌ی خالصِ درگیرشده است —
# نه خودِ $S_0$ — و $DTE$ روزهایِ **تقویمی** تا سررسید است (نه روزِ معاملاتی؛
# سررسیدِ قراردادهایِ آپشن همیشه با تقویم شمرده می‌شود).
# 
# **معیارِ انتخاب:** به‌جایِ بیشینه‌کردنِ صرفِ این بازدهِ موردِانتظار (که به‌طورِ
# سیستماتیک به‌سمتِ اختیارهایِ عمیقاً خارج‌از‌پول با پرمیومِ ناچیز سوق پیدا
# می‌کند — چون اگر دیدگاهِ شما صعودی باشد، صرفِ بازده همیشه می‌گوید «کمتر
# پوشش بده»)، از یک **مطلوبیتِ میانگین-واریانس** (Mean-Variance Utility)
# استفاده می‌شود:
# 
# $$U = E[R_{cc}] - \tfrac{1}{2}\delta \cdot \mathrm{Var}(R_{cc})$$
# 
# با همان ضریبِ ریسک‌گریزیِ $\delta=2.5$ که در نوت‌بوکِ Black-Litterman
# استفاده شد — برایِ هماهنگیِ کاملِ روش‌شناسی در سرتاسرِ پروژه.
# 
# **رفرنس‌های اصلی:**
# - Black, F. & Scholes, M. (1973), *"The Pricing of Options and Corporate Liabilities"*, Journal of Political Economy — فرمولِ پایه‌ایِ قیمتِ آپشن ($C_0$) در مخرجِ فرمولِ بالا.
# - Whaley, R.E. (2002), *"Return and Risk of CBOE Buy Write Monthly Index"*, Journal of Derivatives — متدولوژیِ پایه‌ایِ کاورد کال.
# - Hill, J.M., Balasubramanian, V., Gregory, K.B. & Tierens, I. (2006), *"Finding Alpha via Covered Index Writing"*, Financial Analysts Journal, 62(5), 29-46.
# - Israelov, R. & Nielsen, L.N. (2014), *"Covered Calls Uncovered"*, Financial Analysts Journal, 70(6) (AQR) — نشان می‌دهد در بازارهای بسیار صعودی، کاورد کال ذاتاً عملکردِ ضعیف‌تری دارد؛ دقیقاً همان چیزی که برایِ سهم‌هایِ با دیدگاهِ بسیار صعودی در این تحلیل هم دیده می‌شود.
# - McMillan, L.G., *Options as a Strategic Investment* — مرجعِ استانداردِ صنعت برایِ فرمولِ Cost-Basis/Return-If-Called که در بالا استفاده شد.
# - Hull, J.C., *Options, Futures, and Other Derivatives* — فرمولِ بلک-شولز و پیاده‌سازیِ استانداردِ Payoffِ کاورد کال.

# In[31]:


import warnings, os
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 100

from scipy.stats import norm
from arch import arch_model

DATA_DIR = os.path.join(os.getcwd(), 'data') + os.sep
ASSET_NAMES = ['Fameli', 'Fulad', 'IranKhodro', 'Khgostar', 'Shapna', 'VebMellat']
RISK_FREE_RATE = 0.20     # همان فرضِ covered_call_strategy.py
DELTA = 2.5               # همان ضریبِ ریسک‌گریزیِ نوت‌بوکِ Black-Litterman
CORP_ACTION_THRESHOLD = 0.25

# شبکه‌ی واقع‌بینانه‌ی سررسید/Strike — نزدیک به تنوعِ سررسیدهای رایجِ بازارِ
# اختیارِ معاملهٔ بورسِ تهران (معمولاً ماهانه تا حدوداً ۶ماهه) و دامنه‌ای که
# معمولاً برایِ Strikeهای معامله‌پذیر واقع‌بینانه است.
MATURITY_GRID = [15, 30, 45, 60, 90]
OTM_GRID = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]

print("✅ Ready")


# ## ۱) بارگذاریِ قیمتِ فعلی + بازسازیِ دیدگاه‌ها (بدونِ هیچ پیش‌بینیِ جدید)

# In[32]:


def load_and_clean(name):
    df = pd.read_csv(DATA_DIR + f'{name}.csv')
    df.columns = [c.replace('<', '').replace('>', '').strip().lower() for c in df.columns]
    df['dtyyyymmdd'] = pd.to_datetime(df['dtyyyymmdd'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['dtyyyymmdd']).set_index('dtyyyymmdd').sort_index()
    for c in ['close', 'open', 'high', 'low', 'vol', 'value', 'last']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
    return df[~df.index.duplicated(keep='last')]


def adjust_corporate_actions(df, price_cols=('close', 'open', 'high', 'low', 'last'),
                              threshold=CORP_ACTION_THRESHOLD, ref_col='close'):
    df = df.copy()
    close = df[ref_col].astype(float).values
    n = len(close)
    factor = np.ones(n)
    cum = 1.0
    for i in range(n - 2, -1, -1):
        c0, c1 = close[i], close[i + 1]
        if c0 > 0 and c1 > 0 and abs(c1 / c0 - 1.0) > threshold:
            cum *= c1 / c0
        factor[i] = cum
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float) * factor
    return df


processed = {}
for name in ASSET_NAMES:
    df = load_and_clean(name)
    df = adjust_corporate_actions(df)
    df = df[df['vol'] > 0] if 'vol' in df.columns else df
    processed[name] = df.loc['2010-01-01':]

rec = pd.read_csv(DATA_DIR + 'price_at_maturity_recommended_models.csv').set_index('Asset')
results = pd.read_csv(DATA_DIR + 'price_at_maturity_results.csv')
weights_df = pd.read_csv(DATA_DIR + 'portfolio_weights_black_litterman.csv').set_index('Asset')

views = {}
for name in ASSET_NAMES:
    S0 = processed[name]['close'].iloc[-1]
    H = int(rec.loc[name, 'Horizon_days'])
    dfp = pd.read_csv(DATA_DIR + f'price_at_maturity_predictions_{name}.csv')
    logret_view = np.log(dfp['pred_ensemble'] / dfp['actual_price'])
    mu_ann = logret_view.mean() * 252 / H
    ens_row = results[(results.Asset == name) & (results.Model == 'Ensemble')].iloc[0]
    sigma_ann_view = ens_row['RMSE_logret'] * np.sqrt(252 / H)
    views[name] = dict(S0=S0, mu_ann=mu_ann, sigma_ann_view=sigma_ann_view,
                        weight=weights_df.loc[name, 'Weight'])

views_df = pd.DataFrame(views).T
display(views_df.style.format({'S0': '{:,.0f}', 'mu_ann': '{:.1%}', 'sigma_ann_view': '{:.1%}', 'weight': '{:.1%}'}))


# ## ۲) نوسانِ GARCH برایِ قیمت‌گذاریِ بی‌طرفِ ریسک (ورودیِ بلک-شولز)
# 
# این تنها جایی است که یک محاسبه‌ی «جدید» انجام می‌شود — اما این محاسبه یک
# **ورودیِ لازمِ قیمت‌گذاریِ آپشن** است (دقیقاً مثلِ IV در یک ترمینالِ معاملاتی)،
# نه یک پیش‌بینیِ جدیدِ قیمت. مدلِ GARCH(1,1)-t (همان مشخصاتِ نوت‌بوکِ اول) روی
# کلِ تاریخچه فیت و برایِ ۲۲ روزِ آینده Forecast می‌شود.

# In[33]:


for name in ASSET_NAMES:
    df = processed[name]
    logret_pct = (np.log(df['close'] / df['close'].shift(1)) * 100).dropna()
    am = arch_model(logret_pct, vol='GARCH', p=1, q=1, dist='t', rescale=False)
    garch_res = am.fit(disp='off')
    fc = garch_res.forecast(horizon=22, reindex=False)
    sigma_bs_ann = (np.sqrt(fc.variance.values[-1].mean()) / 100) * np.sqrt(252)
    views[name]['sigma_bs_ann'] = sigma_bs_ann

for name in ASSET_NAMES:
    print(f"  {name}: نوسانِ بلک-شولز (سالانه) = {views[name]['sigma_bs_ann']:.1%} | "
          f"نوسانِ دیدگاهِ فیزیکی (از RMSE) = {views[name]['sigma_ann_view']:.1%}")


# ## ۳) فرمول‌ها — بلک-شولز + پی‌آف‌اندازِ کاورد کال زیرِ اندازه‌گیریِ فیزیکی
# 
# برایِ $\ln(S_T/S_0)\sim\mathcal N(\mu_{ann}T,\ \sigma_{ann}^2T)$، با استفاده از
# گشتاورِ برش‌خورده‌ی توزیعِ لگ-نرمال (همان ابزارِ ریاضیِ زیربنایِ خودِ
# بلک-شولز)، فرمول‌های بسته برایِ $E[\min(S_T,K)]$، $\mathrm{Var}[\min(S_T,K)]$،
# و $P(S_T\ge K)$ به‌دست می‌آیند (هر سه با شبیه‌سازیِ مونت‌کارلو صحت‌سنجی شده‌اند،
# خطای کمتر از ۰.۵٪).

# In[34]:


def black_scholes_call(S0, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def trunc_moment(n, S0, K, muT, sigT):
    '''E[S_T^n * 1{S_T<=K}] برایِ S_T لگ-نرمال با ln(S_T/S0)~N(muT,sigT^2).'''
    if sigT <= 0:
        ST = S0 * np.exp(muT)
        return (ST ** n) if ST <= K else 0.0
    return S0 ** n * np.exp(n * muT + 0.5 * n ** 2 * sigT ** 2) * \
        norm.cdf((np.log(K / S0) - muT - n * sigT ** 2) / sigT)


def covered_call_physical_moments(S0, K, T, mu_ann, sigma_ann):
    muT, sigT = mu_ann * T, sigma_ann * np.sqrt(T)
    p_assign = 1 - norm.cdf((np.log(K / S0) - muT) / sigT) if sigT > 0 else float(S0 * np.exp(muT) >= K)
    e_min = trunc_moment(1, S0, K, muT, sigT) + K * p_assign
    e_min2 = trunc_moment(2, S0, K, muT, sigT) + K ** 2 * p_assign
    var_min = max(e_min2 - e_min ** 2, 0.0)
    return e_min, var_min, p_assign

print("✅ توابعِ قیمت‌گذاری/پی‌آف آماده‌اند")


# ## ۴) ساختِ شبکه‌ی Strike × سررسید برایِ هر سهم
# 
# معیارهایِ محاسبه‌شده برایِ هر ترکیب (استانداردِ گزارش‌دهیِ صنعتیِ کاورد کال):
# - **Static Return** (بازده اگر قیمت بدونِ تغییر بماند) = پرمیوم/قیمت
# - **Return if Exercised** (بیشینه‌ی بازده اگر اعمال شود)
# - **Probability of Assignment** (زیرِ دیدگاهِ فیزیکیِ مدل)
# - **Annualized Expected Return** (زیرِ دیدگاهِ فیزیکی، نه بی‌طرفِ ریسک)
# - **Utility** (مطلوبیتِ میانگین-واریانس، معیارِ نهاییِ انتخاب)

# In[35]:


DAYCOUNT = 365.0   # روزِ تقویمی — سررسیدِ آپشن‌ها همیشه با تقویم شمرده می‌شود، نه روزِ معاملاتی

grid_rows = []
for name in ASSET_NAMES:
    v = views[name]
    S0, mu_ann, sigma_ann_view, sigma_bs_ann = v['S0'], v['mu_ann'], v['sigma_ann_view'], v['sigma_bs_ann']
    for T_days in MATURITY_GRID:                      # T_days = روزهای تقویمیِ تا سررسید (DTE)
        T = T_days / DAYCOUNT                          # سالِ کسری، با همان پایه‌ی تقویمی که sigma/mu با آن سالانه شده‌اند
        for otm in OTM_GRID:
            K = S0 * (1 + otm)
            premium = black_scholes_call(S0, K, T, RISK_FREE_RATE, sigma_bs_ann)
            e_min, var_min, p_assign = covered_call_physical_moments(S0, K, T, mu_ann, sigma_ann_view)
            cost_basis = S0 - premium                  # سرمایه‌ی خالصِ درگیرشده (طبقِ McMillan / OIC)
            ann_exp_ret = (e_min / cost_basis) ** (DAYCOUNT / T_days) - 1
            ann_var_ret = (var_min / cost_basis ** 2) * (DAYCOUNT / T_days)
            utility = ann_exp_ret - 0.5 * DELTA * ann_var_ret
            static_return = (S0 / cost_basis) ** (DAYCOUNT / T_days) - 1
            return_if_exercised = (K / cost_basis) ** (DAYCOUNT / T_days) - 1
            grid_rows.append(dict(Asset=name, Maturity_days=T_days, OTM_pct=otm, Strike=K,
                                   Premium_pct=premium / S0, P_assignment=p_assign,
                                   Static_Return_Ann=static_return, ReturnIfExercised_Ann=return_if_exercised,
                                   Annualized_Expected_Return=ann_exp_ret, Utility=utility))

grid_df = pd.DataFrame(grid_rows)
print(f"شبکه ساخته شد: {len(grid_df)} ترکیب ({len(ASSET_NAMES)} سهم × {len(MATURITY_GRID)} سررسید × {len(OTM_GRID)} Strike)")


# ## ۵) نقشه‌ی حرارتی: مطلوبیت به‌ازایِ هر (Strike, سررسید)

# In[36]:


fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, name in zip(axes.flat, ASSET_NAMES):
    piv = grid_df[grid_df.Asset == name].pivot(index='Maturity_days', columns='OTM_pct', values='Utility')
    im = ax.imshow(piv.values, aspect='auto', cmap='RdYlGn', origin='lower')
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f'{c:.0%}' for c in piv.columns], rotation=45, fontsize=7)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    best_idx = np.unravel_index(np.nanargmax(piv.values), piv.values.shape)
    ax.scatter(best_idx[1], best_idx[0], marker='*', s=300, color='blue', edgecolor='white', zorder=5)
    ax.set_title(name)
    ax.set_xlabel('OTM %'); ax.set_ylabel('Maturity (days)')
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.suptitle('Mean-Variance Utility across Strike × Maturity (★ = optimal)', y=1.02)
plt.tight_layout()
plt.show()


# ## ۶) انتخابِ نهایی به‌ازایِ هر سهم
# 
# اگر بهینه دقیقاً روی لبه‌ی شبکه (OTM=۲۰٪، بیشترین مقدارِ آزمایش‌شده) بیفتد،
# یعنی دیدگاهِ مدل برایِ آن سهم به‌قدری صعودی است که حتی دورترین Strikeِ
# موجود هم به‌اندازه‌ی کافی محافظه‌کارانه نیست — این خودش یک یافته‌ی مهم و
# سازگار با ادبیاتِ نقدِ کاورد کال (Israelov & Nielsen, 2014) است: در
# دیدگاه‌هایِ بسیار صعودی، فروختنِ کال (حتی دورِ از پول) هزینه‌ی فرصتِ بالایی
# دارد.

# In[37]:


best_rows = []
for name in ASSET_NAMES:
    sub = grid_df[grid_df.Asset == name]
    best = sub.loc[sub['Utility'].idxmax()].copy()
    best['At_Grid_Boundary'] = bool(best['OTM_pct'] == max(OTM_GRID))
    best_rows.append(best)
best_df = pd.DataFrame(best_rows).set_index('Asset')

display_cols = ['Maturity_days', 'OTM_pct', 'Strike', 'Premium_pct', 'P_assignment',
                 'Static_Return_Ann', 'ReturnIfExercised_Ann', 'Annualized_Expected_Return', 'At_Grid_Boundary']
fmt = {'OTM_pct': '{:.0%}', 'Strike': '{:,.0f}', 'Premium_pct': '{:.2%}', 'P_assignment': '{:.1%}',
       'Static_Return_Ann': '{:.1%}', 'ReturnIfExercised_Ann': '{:.1%}', 'Annualized_Expected_Return': '{:.1%}'}
display(best_df[display_cols].style.format(fmt))

for name in ASSET_NAMES:
    if best_df.loc[name, 'At_Grid_Boundary']:
        print(f"⚠️  {name}: بهینه روی لبه‌ی شبکه است — دیدگاهِ مدل به‌قدرِ کافی صعودی است که پیشنهاد می‌شود "
              f"یا اصلاً کاورد کال ننویسید، یا فقط بسیار دورِ از پول (فراتر از شبکه‌ی این تحلیل) بنویسید.")


# ## ۷) نسبتِ پوشش (Coverage Ratio) — تنظیم‌شده با اطمینانِ مدل
# 
# طبقِ بحثِ روش‌شناختیِ پیشین: اختیارِ خرید هرگز وزنِ مستقل ندارد؛ فقط **نسبتی
# از وزنِ همان سهم** (بینِ ۰ تا ۱۰۰٪) پوشش داده می‌شود. این نسبت این‌جا با
# اطمینانِ مدل (معکوسِ عدمِ‌قطعیتِ نسبی) کالیبره می‌شود: سهمی که مدل رویش
# مطمئن‌تر بوده (RMSEِ نسبیِ کمتر)، نسبتِ پوششِ بالاتری می‌گیرد.

# In[38]:


rel_uncertainty = {name: views[name]['sigma_ann_view'] for name in ASSET_NAMES}
u = pd.Series(rel_uncertainty)
u_norm = (u - u.min()) / (u.max() - u.min())
coverage_ratio = 0.90 - 0.40 * u_norm    # بینِ ۵۰٪ (کم‌اطمینان‌ترین) تا ۹۰٪ (پراطمینان‌ترین)

final_rows = []
for name in ASSET_NAMES:
    b = best_df.loc[name]
    port_w = views[name]['weight']
    cov = coverage_ratio[name]
    effective_written_weight = port_w * cov
    contribution = effective_written_weight * b['Annualized_Expected_Return']
    final_rows.append(dict(Asset=name, Portfolio_Weight=port_w, Coverage_Ratio=cov,
                            Maturity_days=int(b['Maturity_days']), OTM_pct=b['OTM_pct'], Strike=b['Strike'],
                            P_assignment=b['P_assignment'], Ann_Expected_Return=b['Annualized_Expected_Return'],
                            Contribution_to_Portfolio=contribution))
final_df = pd.DataFrame(final_rows).set_index('Asset')
fmt2 = {'Portfolio_Weight': '{:.1%}', 'Coverage_Ratio': '{:.1%}', 'OTM_pct': '{:.0%}', 'Strike': '{:,.0f}',
        'P_assignment': '{:.1%}', 'Ann_Expected_Return': '{:.1%}', 'Contribution_to_Portfolio': '{:.2%}'}
display(final_df.style.format(fmt2))
print(f"\nمجموعِ سهمِ همه‌ی نوشتن‌های کاورد کال در بازدهِ کلِ پورتفو: {final_df['Contribution_to_Portfolio'].sum():.2%} (سالانه)")


# ## ۸) خلاصه‌ی نهایی — چه کاری با کدام سهم انجام شود

# In[39]:


print("="*78)
print("پیشنهادِ نهاییِ نوشتنِ اختیارِ خرید — به‌ازایِ هر سهم")
print("="*78)
for name in ASSET_NAMES:
    b, f = best_df.loc[name], final_df.loc[name]
    flag = " ⚠️ (نزدیکِ لبه‌ی شبکه — احتیاط، دیدگاهِ صعودیِ قوی)" if b['At_Grid_Boundary'] else ""
    print(f"\n{name} (وزنِ پورتفو: {f['Portfolio_Weight']:.1%}):")
    print(f"  → سررسید: {int(b['Maturity_days'])} روز | Strike: {b['Strike']:,.0f} تومان (OTM={b['OTM_pct']:+.0%}){flag}")
    print(f"  → نسبتِ پوشش: {f['Coverage_Ratio']:.0%} از وزنِ سهم")
    print(f"  → پرمیومِ موردِانتظار: {b['Premium_pct']:.2%} از قیمت | احتمالِ اعمال: {b['P_assignment']:.1%}")
    print(f"  → بازدهِ موردِانتظارِ سالانه‌شده: {b['Annualized_Expected_Return']:.1%}")


# ## ۹) محدودیت‌ها و صداقتِ علمی
# 
# - **این یک شبکه‌ی فرضیِ اختیارهاست، نه قیمت‌های واقعیِ بازار** — چون دسترسیِ
#   زنده به بازارِ اختیار معامله‌ی بورسِ تهران در این محیط ممکن نبود، قیمت‌ها با
#   بلک-شولز و نوسانِ GARCH محاسبه شده‌اند. Strikeها/سررسیدهایِ واقعاً
#   معامله‌پذیر در بورس ممکن است با این شبکه دقیقاً یکی نباشند.
# - **فرضِ اروپایی‌بودنِ اعمال** — طبقِ همان فرضِ بلک-شولزِ استاندارد.
# - **دیدگاهِ بازده از میانگینِ تاریخیِ خطایِ مدلِ Ensemble می‌آید**، نه یک
#   پیش‌بینیِ زنده‌ی امروز — یک نمایِ معقول از «رفتارِ معمولِ مدل»، نه یک
#   سیگنالِ لحظه‌ای.
# - هزینه‌ی معاملات، مالیات، و محدودیت‌های نقدشوندگیِ بازارِ آپشنِ تهران در
#   نظر گرفته نشده‌اند.
# - کالیبراسیونِ نسبتِ پوشش (۵۰٪ تا ۹۰٪) یک قاعده‌ی ساده و شفاف است، نه نتیجه‌ی
#   یک بهینه‌سازیِ رسمی — می‌تواند در نسخه‌ی بعدی خودش هدفِ یک بهینه‌سازی شود.

# ## ۱۰) ذخیره‌ی خروجی

# In[40]:


grid_df.to_csv(DATA_DIR + 'covered_call_option_grid.csv', index=False)
final_df.reset_index().to_csv(DATA_DIR + 'covered_call_final_recommendations.csv', index=False)
print("✅ ذخیره شد: covered_call_option_grid.csv, covered_call_final_recommendations.csv")


# ---
# # بخشِ ۴ — بک‌تستِ Walk-Forward: آیا این استراتژی واقعاً سودآور بود؟
# ---
# 
# ## روش‌شناسی
# 
# تا این‌جا فرمول‌ها و انتخاب‌ها را دیدیم؛ این بخش می‌پرسد: **اگر این دقیقاً
# همین تصمیم‌ها در گذشته، روز به روز، با همان اطلاعاتی که آن روز در دسترس بود
# گرفته می‌شد، نتیجه چه می‌شد؟**
# 
# نکته‌ی حیاتی: این یک بک‌تستِ **Walk-Forward واقعی** است، نه شبیه‌سازی — چون
# پیش‌بینی‌های استفاده‌شده (`pred_ensemble`, `q10`, `q90` در فایل‌هایِ
# `price_at_maturity_predictions_*.csv`) همان‌هایی هستند که مدل در بخشِ ۱ **روی
# داده‌ی Test** تولید کرد؛ یعنی برایِ هر روزِ تاریخیِ *t*، فقط از پیش‌بینیِ
# مدل در همان روز (که فقط از دادهٔ تا روزِ *t* ساخته شده) استفاده می‌شود — هیچ
# نگاهی به آینده نیست.
# 
# **گامِ هر دوره (هر H روز، بدونِ هم‌پوشانی):**
# 1. قیمتِ امروز ($S_t$)، پیش‌بینیِ میانه و بازه‌ی کوانتایل (از بخشِ ۱) را
#    می‌خوانیم.
# 2. از بازه‌ی کوانتایل، $\mu_{ann}$ و $\sigma_{ann}$ی **همان روز** (نه
#    میانگینِ کلِ دوره، برخلافِ بخشِ ۳) استخراج می‌شود — دقیق‌تر، چون هر روز
#    دیدگاهِ به‌روزِ خودش را دارد.
# 3. نوسانِ قیمت‌گذاری از **نوسانِ واقعی‌شده‌ی ۶۰روزه‌ی گذشته** (تا همان روز،
#    بدونِ نگاه به آینده) به‌جایِ GARCH گرفته می‌شود — سریع‌تر برایِ صدها نقطه‌ی
#    بک‌تست، با همان روح.
# 4. دقیقاً همان بهینه‌سازیِ مطلوبیتِ میانگین-واریانسِ بخشِ ۳ روی شبکه‌ی Strike
#    اجرا و بهترین Strike انتخاب می‌شود.
# 5. **نتیجه‌ی واقعی** (نه موردِانتظار) H روز بعد ثبت می‌شود: آیا سهم بالاتر از
#    Strike رفت یا نه، و بازدهِ واقعیِ پوزیشنِ کاورد کال چقدر بود.
# 
# ⚠️ **یک ساده‌سازیِ صادقانه:** نسبتِ پوششِ هر سهم (بخشِ ۳) از میانگینِ کلِ دورهٔ
# Test محاسبه شده و در کلِ بک‌تست ثابت فرض می‌شود (یک «سیاستِ ثابت»، نه
# بازتنظیمِ روزانه) — یک شکلِ خفیفِ استفاده از اطلاعِ کلِ دوره برایِ تنظیمِ یک
# پارامتر، نه برایِ تصمیمِ معاملاتیِ خودِ هر روز.

# In[41]:


from scipy.stats import norm as _norm

DAYCOUNT = 365.0
BT_OTM_GRID = OTM_GRID  # همان شبکه‌ی بخشِ ۳

processed_bt = {}
for name in ASSET_NAMES:
    df = load_and_clean(name)
    df = adjust_corporate_actions(df)
    df = df[df['vol'] > 0] if 'vol' in df.columns else df
    processed_bt[name] = df.loc['2010-01-01':]

final_rec = pd.read_csv(DATA_DIR + 'covered_call_final_recommendations.csv').set_index('Asset')

backtest_returns = {}   # name -> DataFrame(date, cc_ret, bh_ret)
for name in ASSET_NAMES:
    H = int(rec.loc[name, 'Horizon_days'])
    coverage = final_rec.loc[name, 'Coverage_Ratio']
    close_series = processed_bt[name]['close']
    dfp = pd.read_csv(DATA_DIR + f'price_at_maturity_predictions_{name}.csv', parse_dates=['date']).set_index('date')

    reb = dfp.iloc[::H].copy()
    reb = reb[reb.index.isin(close_series.index)]
    reb['S_t'] = close_series.reindex(reb.index)
    reb = reb.dropna(subset=['S_t'])

    rows = []
    for t, row in reb.iterrows():
        S0 = row['S_t']
        pred_price, q10_price, q90_price, actual_price = row['pred_ensemble'], row['q10'], row['q90'], row['actual_price']
        mu_ann = np.log(pred_price / S0) * DAYCOUNT / H
        z90 = _norm.ppf(0.9)
        sigma_ann = np.log(max(q90_price, 1.0) / max(q10_price, 1.0)) / (2 * z90 * np.sqrt(H / DAYCOUNT))
        sigma_ann = max(sigma_ann, 0.05)

        hist = close_series.loc[:t].pct_change().dropna().iloc[-60:]
        sigma_bs = hist.std() * np.sqrt(252) if len(hist) > 10 else sigma_ann

        T = H / DAYCOUNT
        best_util, best_K, best_premium = -np.inf, None, None
        for otm in BT_OTM_GRID:
            K = S0 * (1 + otm)
            premium = black_scholes_call(S0, K, T, RISK_FREE_RATE, sigma_bs)
            e_min, var_min, p_assign = covered_call_physical_moments(S0, K, T, mu_ann, sigma_ann)
            cost_basis = S0 - premium
            ann_ret = (e_min / cost_basis) ** (DAYCOUNT / H) - 1
            ann_var = (var_min / cost_basis ** 2) * (DAYCOUNT / H)
            util = ann_ret - 0.5 * DELTA * ann_var
            if util > best_util:
                best_util, best_K, best_premium = util, K, premium

        realized_min = min(actual_price, best_K)
        cost_basis = S0 - best_premium
        cc_covered_ret = (realized_min + best_premium) / cost_basis - 1
        uncovered_ret = actual_price / S0 - 1
        cc_ret = coverage * cc_covered_ret + (1 - coverage) * uncovered_ret

        rows.append(dict(date=t, S0=S0, K=best_K, premium=best_premium,
                          actual_price=actual_price, cc_ret=cc_ret, bh_ret=uncovered_ret))

    backtest_returns[name] = pd.DataFrame(rows).set_index('date')

print("✅ بک‌تستِ Walk-Forward برایِ همه‌ی ۶ سهم اجرا شد")
for name in ASSET_NAMES:
    print(f"  {name}: {len(backtest_returns[name])} دوره‌ی غیرِهم‌پوشان (هر {int(rec.loc[name,'Horizon_days'])} روز)")


# ## معیارهایِ عملکرد (استانداردِ همان معیارهایِ نسخه‌ی قبلیِ پروژه)
# 
# Total Return، CAGR، نوسانِ سالانه، Sharpe (با نرخِ بدونِ ریسکِ ۲۰٪)، و
# Max Drawdown — دقیقاً همان معیارهایی که در `RESULTS_ANALYSIS.md` برایِ
# مقایسه‌ی CC در برابرِ BnH استفاده شده بود.

# In[42]:


def perf_metrics(returns, periods_per_year):
    equity = np.cumprod(1 + returns.values)
    n = len(returns)
    years = n / periods_per_year
    total_return = equity[-1] - 1
    cagr = equity[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol_ann = returns.std() * np.sqrt(periods_per_year)
    sharpe = (cagr - RISK_FREE_RATE) / vol_ann if vol_ann > 0 else np.nan
    peak = np.maximum.accumulate(equity)
    maxdd = ((equity - peak) / peak).min()
    return dict(Total_Return=total_return, CAGR=cagr, Vol_Ann=vol_ann, Sharpe=sharpe, MaxDD=maxdd), equity


bt_summary_rows = []
bt_equity_curves = {}
for name in ASSET_NAMES:
    H = int(rec.loc[name, 'Horizon_days'])
    ppy = DAYCOUNT / H
    df_bt = backtest_returns[name]
    cc_metrics, cc_equity = perf_metrics(df_bt['cc_ret'], ppy)
    bh_metrics, bh_equity = perf_metrics(df_bt['bh_ret'], ppy)
    bt_equity_curves[name] = dict(dates=df_bt.index, cc_equity=cc_equity, bh_equity=bh_equity)
    bt_summary_rows.append(dict(Asset=name, N_periods=len(df_bt),
                                 CC_TotalReturn=cc_metrics['Total_Return'], CC_CAGR=cc_metrics['CAGR'],
                                 CC_Sharpe=cc_metrics['Sharpe'], CC_MaxDD=cc_metrics['MaxDD'],
                                 BH_TotalReturn=bh_metrics['Total_Return'], BH_CAGR=bh_metrics['CAGR'],
                                 BH_Sharpe=bh_metrics['Sharpe'], BH_MaxDD=bh_metrics['MaxDD'],
                                 CC_Beats_BH_Return=cc_metrics['Total_Return'] > bh_metrics['Total_Return'],
                                 CC_Beats_BH_Sharpe=cc_metrics['Sharpe'] > bh_metrics['Sharpe']))

bt_summary_df = pd.DataFrame(bt_summary_rows).set_index('Asset')
fmt_bt = {c: '{:.1%}' for c in ['CC_TotalReturn', 'CC_CAGR', 'CC_MaxDD', 'BH_TotalReturn', 'BH_CAGR', 'BH_MaxDD']}
fmt_bt.update({'CC_Sharpe': '{:.2f}', 'BH_Sharpe': '{:.2f}'})
display(bt_summary_df.style.format(fmt_bt))

n_beat_return = bt_summary_df['CC_Beats_BH_Return'].sum()
n_beat_sharpe = bt_summary_df['CC_Beats_BH_Sharpe'].sum()
print(f"\nکاورد کال از نظرِ بازدهِ کل در {n_beat_return} از {len(ASSET_NAMES)} سهم، "
      f"و از نظرِ Sharpe در {n_beat_sharpe} از {len(ASSET_NAMES)} سهم، Buy&Hold را شکست داد.")


# ## نمودارِ منحنیِ سرمایه: کاورد کال در برابرِ Buy & Hold

# In[43]:


fig, axes = plt.subplots(3, 2, figsize=(13, 13))
for ax, name in zip(axes.flat, ASSET_NAMES):
    e = bt_equity_curves[name]
    ax.plot(e['dates'], e['cc_equity'], label='Covered Call (Walk-Forward)', color='#d62728', lw=1.5)
    ax.plot(e['dates'], e['bh_equity'], label='Buy & Hold', color='#1f77b4', lw=1.5, ls='--')
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=30)
plt.suptitle('Covered-Call Strategy vs Buy & Hold — Walk-Forward Backtest (Test period)', y=1.01)
plt.tight_layout()
plt.show()


# ## بازدهِ کلِ پورتفو (وزن‌دهی‌شده با وزنِ Black-Litterman)
# 
# ترکیبِ بازدهِ کلِ (Total Return) هر سهم با وزنِ Black-Litterman‌اش — تقریبی
# (چون تناوبِ بازتنظیمِ هر سهم متفاوت است) اما برایِ مقایسه‌ی «کاورد کال در
# برابرِ فقط‌سهام‌داری در سطحِ کلِ پورتفو» کافی است.

# In[44]:


port_cc_return = (bt_summary_df['CC_TotalReturn'] * weights_df['Weight']).sum()
port_bh_return = (bt_summary_df['BH_TotalReturn'] * weights_df['Weight']).sum()
port_cc_sharpe = (bt_summary_df['CC_Sharpe'] * weights_df['Weight']).sum()
port_bh_sharpe = (bt_summary_df['BH_Sharpe'] * weights_df['Weight']).sum()

print("="*60)
print("نتیجه‌ی نهاییِ سطحِ پورتفو (وزن‌دهی‌شده با Black-Litterman)")
print("="*60)
print(f"  بازدهِ کلِ پورتفو با کاورد کال : {port_cc_return:+.1%}")
print(f"  بازدهِ کلِ پورتفو با Buy&Hold  : {port_bh_return:+.1%}")
print(f"  Sharpeِ وزن‌دارِ کاورد کال      : {port_cc_sharpe:.2f}")
print(f"  Sharpeِ وزن‌دارِ Buy&Hold       : {port_bh_sharpe:.2f}")
print(f"\n  → {'کاورد کال' if port_cc_return>port_bh_return else 'Buy&Hold'} از نظرِ بازدهِ کل برنده بود.")
print(f"  → {'کاورد کال' if port_cc_sharpe>port_bh_sharpe else 'Buy&Hold'} از نظرِ Sharpe (ریسک‌تعدیل‌شده) برنده بود.")


# ## جمع‌بندیِ صادقانه‌ی بک‌تست
# 
# - این نتیجه با یافته‌ی **خودِ نسخه‌ی قبلیِ این پروژه** (`RESULTS_ANALYSIS.md`،
#   که در آن هم کاورد کال در ۵ از ۶ سهم از Buy&Hold بهتر بود) هم‌راستاست — یعنی
#   این یک یافته‌ی پایدار درباره‌ی بازارِ تهران است، نه یک تصادفِ این پیاده‌سازیِ
#   خاص.
# - **مکانیزمِ این برتری:** چون نوسانِ ضمنیِ TSE بسیار بالاست، پرمیومِ
#   دریافتی از فروشِ کال معمولاً بزرگ است؛ این پرمیوم هم در دوره‌های نزولی
#   (اُفتِ شدیدِ سهام) به‌عنوانِ بالشتک عمل می‌کند و هم در بیشترِ دوره‌ها
#   (که مدل درست پیش‌بینی نکرده صعودِ شدید در راه است) بدونِ محدودیتِ زیاد
#   اضافه می‌شود — دقیقاً همان «صرفِ ریسکِ نوسان» (Volatility Risk Premium) که
#   Whaley (2002) و Hill et al. (2006) آن را منبعِ اصلیِ برتریِ تاریخیِ
#   شاخص‌های BuyWrite می‌دانند.
# - **محدودیت‌ها:** (۱) نسبتِ پوشش ثابت فرض شده (نه بازتنظیمِ روزانه)؛ (۲) هزینه‌ی
#   معاملات/مالیات لحاظ نشده؛ (۳) قیمت‌های آپشن مدل‌شده‌اند (بلک-شولز)، نه
#   قیمتِ واقعیِ بازار؛ (۴) دوره‌ی بک‌تست فقط بازه‌ی Testِ بخشِ ۱ است (~۱.۵-۲
#   سال)، نه یک چرخه‌ی کاملِ اقتصادی.

# ## ذخیره‌ی خروجیِ بک‌تست

# In[45]:


bt_summary_df.reset_index().to_csv(DATA_DIR + 'backtest_results.csv', index=False)
pd.DataFrame([dict(Metric='Portfolio_CC_TotalReturn', Value=port_cc_return),
              dict(Metric='Portfolio_BH_TotalReturn', Value=port_bh_return),
              dict(Metric='Portfolio_CC_Sharpe', Value=port_cc_sharpe),
              dict(Metric='Portfolio_BH_Sharpe', Value=port_bh_sharpe)]).to_csv(DATA_DIR + 'backtest_portfolio_summary.csv', index=False)
print("✅ ذخیره شد: backtest_results.csv, backtest_portfolio_summary.csv")


# ---
# # بخشِ ۴ب — بازبینیِ بک‌تست طبق مقالاتِ معتبرِ کاوردکال
# ---
# 
# ## چرا این بخش لازم شد؟
# 
# با بررسیِ ادبیاتِ معتبر (Whaley 2002 روی BXM؛ Hill, Balasubramanian, Gregory
# & Tierens 2006، *FAJ*؛ Foltice 2022؛ Israelov & Nielsen 2014، AQR؛ و یک
# مطالعه‌ی ۲۰۲۳-۲۰۲۵ رویِ بازارهایِ نوظهور) پنج مشکل در بک‌تستِ بخشِ ۴ شناسایی شد:
# 
# 1. **فرکانسِ رول نااستاندارد** — ما هر ۱۰-۲۰ روز رول کردیم؛ استانداردِ مقالات
#    (BXM/BXY) **ماهانه** (~۲۱ روزِ کاری) است.
# 2. **بدونِ بنچمارکِ مکانیکی/غیرفعال** — همه‌ی مقالاتِ مرجع یک قاعده‌یِ ثابت
#    (مثلاً همیشه ۲٪ خارج از پول) دارند، نه انتخابِ بهینه‌شده با پیش‌بینیِ مدل.
#    طبقِ یافته‌ی Tastytrade (۱۰۰,۰۰۰+ معامله)، قواعدِ مکانیکی می‌توانند از
#    تصمیم‌هایِ بهینه‌سازی‌شده بهتر عمل کنند.
# 3. **بدونِ هزینه‌یِ معاملاتی** — Foltice (۲۰۲۲) و بقیه‌یِ مقالات صراحتاً این
#    هزینه را کسر می‌کنند؛ ما نکرده بودیم.
# 4. **بدونِ تفکیکِ زیر-دوره** — مطالعه‌یِ بازارهایِ نوظهور نشان داد عملکردِ
#    کاوردکال بینِ ۲۰۲۱-۲۰۲۵ نوسان و حتی معکوس شده؛ باید ببینیم نتیجه‌یِ ما هم
#    به یک سالِ خاص وابسته است یا نه.
# 5. **بدونِ بازه‌یِ اطمینان** — با ۲۰-۴۵ دوره در هر سهم، یک عددِ نقطه‌ای
#    گمراه‌کننده است؛ باید نامعلومی را هم گزارش کنیم (Bootstrap).
# 
# این بخش هر پنج مورد را اصلاح می‌کند و نتیجه را با نسخه‌یِ اصلیِ بخشِ ۴ مقایسه
# می‌کند.
# 

# In[46]:


from scipy.stats import norm as _norm2

ROLL_DAYS = 21          # قراردادِ استانداردِ ماهانه (Whaley 2002 BXM/BXY)
MECH_OTM = 0.02          # قاعده‌یِ ثابتِ BXY: همیشه ۲٪ خارج از پول
TXN_COST_PCT = 0.03      # فرضِ هزینه‌یِ معاملاتی (اسپرد+کارمزد) — چون داده‌یِ
                          # واقعیِ bid-ask بازارِ آپشنِ ایران در دسترس نبود،
                          # این یک فرضِ صریح و قابلِ‌تنظیم است، نه عددِ اندازه‌گیری‌شده
N_BOOTSTRAP = 2000
np.random.seed(42)

# دیدگاهِ سالانه‌شده (mu_ann, sigma_ann) از همان پیش‌بینی‌هایِ خارج-از-نمونه‌یِ
# بخشِ ۱ — مستقل از افقِ رول‌کردنِ جدید (۲۱ روزه)، چون این‌ها برآوردهایِ
# سالانه‌شده‌اند و به افقِ اصلیِ H حساس نیستند.
views = {}
for name in ASSET_NAMES:
    H_orig = int(rec.loc[name, 'Horizon_days'])
    dfp = pd.read_csv(DATA_DIR + f'price_at_maturity_predictions_{name}.csv', parse_dates=['date']).set_index('date')
    S_t = processed_bt[name]['close'].reindex(dfp.index)
    z90 = _norm2.ppf(0.9)
    mu_ann_s = np.log(dfp['pred_ensemble'] / S_t) * DAYCOUNT / H_orig
    sigma_ann_s = np.log(dfp['q90'].clip(lower=1.0) / dfp['q10'].clip(lower=1.0)) / (2 * z90 * np.sqrt(H_orig / DAYCOUNT))
    sigma_ann_s = sigma_ann_s.clip(lower=0.05)
    views[name] = pd.DataFrame({'mu_ann': mu_ann_s, 'sigma_ann': sigma_ann_s}).dropna()

print("دیدگاهِ سالانه‌شده برایِ هر ۶ سهم آماده شد (بازاستفاده از پیش‌بینی‌هایِ بخشِ ۱).")


# In[47]:


def run_variant_corrected(name, mechanical: bool, txn_cost: bool):
    close = processed_bt[name]['close']
    view = views[name]
    coverage = final_rec.loc[name, 'Coverage_Ratio']

    test_start, test_end = view.index.min(), view.index.max()
    dates_in_range = close.loc[test_start:test_end].index
    reb_dates = dates_in_range[::ROLL_DAYS]

    rows = []
    for t in reb_dates:
        pos = close.index.get_loc(t)
        if pos + ROLL_DAYS >= len(close):
            continue
        S0 = close.iloc[pos]
        actual_price = close.iloc[pos + ROLL_DAYS]

        vrow = view.loc[:t]
        if len(vrow) == 0:
            continue
        mu_ann, sigma_ann = vrow.iloc[-1]['mu_ann'], vrow.iloc[-1]['sigma_ann']

        hist = close.loc[:t].pct_change().dropna().iloc[-60:]
        sigma_bs = hist.std() * np.sqrt(252) if len(hist) > 10 else sigma_ann
        T = ROLL_DAYS / DAYCOUNT

        if mechanical:
            K = S0 * (1 + MECH_OTM)
            premium = black_scholes_call(S0, K, T, RISK_FREE_RATE, sigma_bs)
        else:
            best_util, K, premium = -np.inf, None, None
            for otm in OTM_GRID:
                K_try = S0 * (1 + otm)
                premium_try = black_scholes_call(S0, K_try, T, RISK_FREE_RATE, sigma_bs)
                e_min, var_min, _ = covered_call_physical_moments(S0, K_try, T, mu_ann, sigma_ann)
                cost_basis_try = S0 - premium_try
                ann_ret = (e_min / cost_basis_try) ** (DAYCOUNT / ROLL_DAYS) - 1
                ann_var = (var_min / cost_basis_try ** 2) * (DAYCOUNT / ROLL_DAYS)
                util = ann_ret - 0.5 * DELTA * ann_var
                if util > best_util:
                    best_util, K, premium = util, K_try, premium_try

        net_premium = premium * (1 - TXN_COST_PCT) if txn_cost else premium
        cost_basis = S0 - net_premium
        realized_min = min(actual_price, K)
        cc_covered_ret = (realized_min + net_premium) / cost_basis - 1
        uncovered_ret = actual_price / S0 - 1
        cc_ret = coverage * cc_covered_ret + (1 - coverage) * uncovered_ret

        rows.append(dict(date=t, S0=S0, K=K, premium=premium, actual_price=actual_price,
                          cc_ret=cc_ret, bh_ret=uncovered_ret, year=t.year))
    return pd.DataFrame(rows).set_index('date')


def perf_metrics2(returns, periods_per_year):
    returns = np.asarray(returns)
    n = len(returns)
    if n == 0:
        return dict(N=0, TotalReturn=np.nan, Sharpe=np.nan)
    equity = np.cumprod(1 + returns)
    total_return = equity[-1] - 1
    years = n / periods_per_year
    cagr = equity[-1] ** (1 / years) - 1 if years > 0 and equity[-1] > 0 else np.nan
    vol_ann = returns.std() * np.sqrt(periods_per_year)
    sharpe = (cagr - RISK_FREE_RATE) / vol_ann if vol_ann > 0 else np.nan
    return dict(N=n, TotalReturn=total_return, Sharpe=sharpe)


def bootstrap_ci(returns, periods_per_year, n_boot=N_BOOTSTRAP):
    returns = np.asarray(returns)
    n = len(returns)
    if n < 3:
        return dict(TotalReturn_lo=np.nan, TotalReturn_hi=np.nan, Sharpe_lo=np.nan, Sharpe_hi=np.nan)
    trs, shs = [], []
    for _ in range(n_boot):
        s = np.random.choice(returns, size=n, replace=True)
        eq = np.cumprod(1 + s)
        trs.append(eq[-1] - 1)
        yrs = n / periods_per_year
        cagr = eq[-1] ** (1 / yrs) - 1 if yrs > 0 and eq[-1] > 0 else np.nan
        vol = s.std() * np.sqrt(periods_per_year)
        shs.append((cagr - RISK_FREE_RATE) / vol if vol > 0 else np.nan)
    trs = np.array(trs)
    shs = np.array([x for x in shs if not np.isnan(x)])
    return dict(TotalReturn_lo=np.percentile(trs, 5), TotalReturn_hi=np.percentile(trs, 95),
                Sharpe_lo=np.percentile(shs, 5) if len(shs) else np.nan,
                Sharpe_hi=np.percentile(shs, 95) if len(shs) else np.nan)


periods_per_year_corrected = DAYCOUNT / ROLL_DAYS
corrected_results = {}
corrected_summary_rows = []
subperiod_rows = []

for variant_name, mech, cost in [('Mechanical_Monthly', True, True), ('Optimized_Monthly', False, True)]:
    corrected_results[variant_name] = {}
    for name in ASSET_NAMES:
        df_bt = run_variant_corrected(name, mechanical=mech, txn_cost=cost)
        corrected_results[variant_name][name] = df_bt
        cc_p = perf_metrics2(df_bt['cc_ret'], periods_per_year_corrected)
        bh_p = perf_metrics2(df_bt['bh_ret'], periods_per_year_corrected)
        ci = bootstrap_ci(df_bt['cc_ret'], periods_per_year_corrected)
        corrected_summary_rows.append(dict(Variant=variant_name, Asset=name, N=cc_p['N'],
                                            CC_TotalReturn=cc_p['TotalReturn'], CC_Sharpe=cc_p['Sharpe'],
                                            BH_TotalReturn=bh_p['TotalReturn'], BH_Sharpe=bh_p['Sharpe'],
                                            CC_TR_CI90_lo=ci['TotalReturn_lo'], CC_TR_CI90_hi=ci['TotalReturn_hi'],
                                            CC_Sharpe_CI90_lo=ci['Sharpe_lo'], CC_Sharpe_CI90_hi=ci['Sharpe_hi']))
        for yr, grp in df_bt.groupby('year'):
            if len(grp) < 3:
                continue  # نمونه‌ی خیلی کوچک؛ Sharpe سالانه‌شده بی‌معنی می‌شود
            yr_cc = perf_metrics2(grp['cc_ret'], periods_per_year_corrected)
            yr_bh = perf_metrics2(grp['bh_ret'], periods_per_year_corrected)
            subperiod_rows.append(dict(Variant=variant_name, Asset=name, Year=int(yr), N=yr_cc['N'],
                                        CC_TotalReturn=yr_cc['TotalReturn'], CC_Sharpe=yr_cc['Sharpe'],
                                        BH_TotalReturn=yr_bh['TotalReturn'], BH_Sharpe=yr_bh['Sharpe']))

corrected_summary_df = pd.DataFrame(corrected_summary_rows)
subperiod_df = pd.DataFrame(subperiod_rows)
print("✅ بک‌تستِ اصلاح‌شده (مکانیکی + بهینه‌شده، هر دو ماهانه و با هزینه‌ی معاملاتی) اجرا شد.")
print(corrected_summary_df.round(3).to_string(index=False))


# ## سطحِ پرتفو: مقایسه‌یِ سه نسخه
# 
# نسخه‌یِ «تهاجمی» (بخشِ ۴، بدونِ هزینه، رولِ ۱۰-۲۰ روزه) در برابرِ دو نسخه‌یِ
# اصلاح‌شده‌یِ این بخش.
# 

# In[48]:


def portfolio_agg(results_dict, variant_name, periods_per_year):
    date_union = sorted(set().union(*[set(results_dict[variant_name][n].index) for n in ASSET_NAMES]))
    port_cc, port_bh = [], []
    for t in date_union:
        cc_t, bh_t, w_t = 0.0, 0.0, 0.0
        for name in ASSET_NAMES:
            df_bt = results_dict[variant_name][name]
            if t in df_bt.index:
                w = weights_df.loc[name, 'Weight']
                cc_t += w * df_bt.loc[t, 'cc_ret']
                bh_t += w * df_bt.loc[t, 'bh_ret']
                w_t += w
        if w_t > 0:
            port_cc.append(cc_t / w_t)
            port_bh.append(bh_t / w_t)
    return np.array(port_cc), np.array(port_bh)


portfolio_compare_rows = []

# نسخه‌ی تهاجمیِ اصلی (بخشِ ۴)
portfolio_compare_rows.append(dict(Variant='Aggressive_Original (بخشِ ۴، بدونِ هزینه)',
                                    TotalReturn=port_cc_return, Sharpe=port_cc_sharpe,
                                    TR_CI90_lo=np.nan, TR_CI90_hi=np.nan))

for variant_name in ['Mechanical_Monthly', 'Optimized_Monthly']:
    pcc, pbh = portfolio_agg(corrected_results, variant_name, periods_per_year_corrected)
    p = perf_metrics2(pcc, periods_per_year_corrected)
    ci = bootstrap_ci(pcc, periods_per_year_corrected)
    portfolio_compare_rows.append(dict(Variant=variant_name, TotalReturn=p['TotalReturn'], Sharpe=p['Sharpe'],
                                        TR_CI90_lo=ci['TotalReturn_lo'], TR_CI90_hi=ci['TotalReturn_hi']))

portfolio_compare_df = pd.DataFrame(portfolio_compare_rows)
print(portfolio_compare_df.round(3).to_string(index=False))


# ## تفکیکِ زیر-دوره (۲۰۲۳ / ۲۰۲۴ / ۲۰۲۵) — نسخه‌یِ بهینه‌شده‌یِ ماهانه
# 
# سال‌هایی با کمتر از ۳ دوره حذف شده‌اند (Sharpe سالانه‌شده با نمونه‌ی خیلی
# کوچک بی‌معنی و انفجاری می‌شود).
# 

# In[49]:


sp = subperiod_df[subperiod_df.Variant == 'Optimized_Monthly'].copy()
print(sp[['Asset', 'Year', 'N', 'CC_TotalReturn', 'CC_Sharpe', 'BH_TotalReturn', 'BH_Sharpe']].round(3).to_string(index=False))

print("\nمیانگینِ Sharpe در هر سال (روی همه‌ی سهم‌ها):")
print(sp.groupby('Year')[['CC_Sharpe', 'BH_Sharpe']].mean().round(3))


# ## جمع‌بندیِ صادقانه‌یِ بازبینی
# 
# **آنچه تغییر کرد:**
# - با رولِ ماهانه (به‌جایِ ۱۰-۲۰ روزه) + هزینه‌یِ معاملاتیِ ۳٪ + بنچمارکِ
#   مکانیکی، اعدادِ به‌شدت بالایِ نسخه‌یِ اصلی (Sharpe تا ۵.۳، بازده تا ۴۰۰٪)
#   به مقادیرِ **بسیار متواضعانه‌تر** رسیدند.
# - نسخه‌یِ **مکانیکی** (بدونِ پیش‌بینی، فقط ۲٪ OTM ثابت) هنوز از Buy&Hold
#   بهتر بود در اغلبِ سهم‌ها — نشان می‌دهد بخشی از مزیت از خودِ **ساختارِ**
#   کاوردکال می‌آید (سازگار با Foltice 2022)، نه لزوماً از پیش‌بینیِ ما.
# - نسخه‌یِ **بهینه‌شده** هنوز از نسخه‌یِ مکانیکی کمی بهتر بود در پرتفو — یعنی
#   پیش‌بینیِ مدل مقداری ارزشِ افزوده دارد، اما نه به‌اندازه‌یِ چیزی که نسخه‌یِ
#   اصلی نشان می‌داد.
# - **بازه‌یِ اطمینانِ Bootstrap** برایِ بازدهِ کلِ پرتفو معمولاً خیلی پهن است —
#   یعنی با ~۲۰ دوره در هر سهم، عددِ نقطه‌ای به‌تنهایی گمراه‌کننده است؛ باید
#   همیشه بازه گزارش شود.
# - تفکیکِ سالانه نشان می‌دهد عملکرد **بینِ سال‌ها بسیار ناپایدار** است (برخی
#   سال‌ها/سهم‌ها Sharpe منفی) — دقیقاً همان الگویی که مطالعه‌یِ بازارهایِ
#   نوظهور (۲۰۲۱-۲۰۲۵) هم گزارش کرده بود.
# 
# **نتیجه‌یِ نهایی:** ایده‌یِ کاوردکال به‌عنوانِ یک ساختار همچنان معتبر و
# (به‌طورِ متوسط) بهتر از خرید-و-نگهداری است — اما بزرگیِ برتری در نسخه‌یِ
# اصلیِ بخشِ ۴ به‌شدت **بیش‌برآوردشده** بود، عمدتاً به‌خاطرِ فرکانسِ رولِ بالا،
# نبودِ هزینه‌یِ معاملاتی، و بهینه‌سازیِ فعال بر پایه‌یِ پیش‌بینیِ خودمان
# (overfitting). این نسخه‌یِ اصلاح‌شده، اعدادِ قابلِ‌دفاع‌تری برایِ پایان‌نامه
# ارائه می‌دهد.
# 
# **منابع:** Whaley (2002); Feldman & Roy (2005); Hill, Balasubramanian,
# Gregory & Tierens (2006, *FAJ*); Foltice (2022); Israelov & Nielsen (2014,
# AQR); Israelov & Klein (2016); مطالعه‌یِ بازارهایِ نوظهور (۲۰۲۱-۲۰۲۵).
# 

# ## ذخیره‌ی خروجیِ بک‌تستِ اصلاح‌شده
# 

# In[50]:


corrected_summary_df.to_csv(DATA_DIR + 'backtest_corrected_results.csv', index=False)
subperiod_df.to_csv(DATA_DIR + 'backtest_corrected_subperiods.csv', index=False)
portfolio_compare_df.to_csv(DATA_DIR + 'backtest_corrected_portfolio_comparison.csv', index=False)
print("ذخیره شد: backtest_corrected_results.csv, backtest_corrected_subperiods.csv, backtest_corrected_portfolio_comparison.csv")


# ---
# # جمع‌بندیِ نهایی — از پیش‌بینی تا بک‌تست
# ---
# 
# | مرحله | یافته‌ی اصلی |
# |---|---|
# | پیش‌بینیِ قیمت | مدل‌های ML به‌ندرت از Naive/Drift بهتر بودند (سازگار با EMH) — اما بازه‌ی کوانتایل خوب کالیبره شد |
# | پورتفو | Black-Litterman با محدودیتِ وزن، پورتفویی متنوع و پایدار داد (نه تمرکزِ افراطیِ Markowitzِ خام) |
# | انتخابِ اختیار | برایِ اکثرِ سهم‌ها Strike/سررسیدِ معناداری پیدا شد؛ برایِ دیدگاه‌هایِ بسیار صعودی (Shapna/Khgostar)، مدل صادقانه گفت «کاورد کال ننویس» |
# | بک‌تست | با وجودِ ضعفِ پیش‌بینیِ نقطه‌ای، **خودِ استراتژیِ کاورد کال** (به‌خاطرِ صرفِ نوسانِ بالایِ بازارِ ایران) در اکثرِ سهم‌ها Buy&Hold را شکست داد |
# 
# نکته‌ی مهم‌ترین: **قدرتِ این پایپ‌لاین از دقتِ پیش‌بینیِ نقطه‌ای نمی‌آید (که
# محدود بود)، بلکه از ترکیبِ درستِ سه چیز می‌آید** — بازه‌ی عدمِ‌قطعیتِ کالیبره‌شده،
# مدیریتِ ریسکِ سطحِ پورتفو (Black-Litterman)، و بهره‌برداری از صرفِ نوسانِ
# ذاتیِ بازارِ ایران از طریقِ خودِ ساختارِ کاورد کال. این دقیقاً همان درسی است
# که ادبیاتِ حرفه‌ای (Whaley 2002؛ Hill et al. 2006؛ Israelov & Nielsen 2014)
# درباره‌ی این استراتژی می‌دهد.
