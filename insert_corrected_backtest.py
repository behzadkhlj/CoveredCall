import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

NB_PATH = '/home/user/CoveredCall/final_covered_call_pipeline.ipynb'
nb = nbformat.read(NB_PATH, as_version=4)


def md(src):
    return new_markdown_cell(src.strip() + "\n")


def code(src):
    return new_code_cell(src.strip() + "\n")


new_cells = []

new_cells.append(md(r"""
---
# بخشِ ۴ب — بازبینیِ بک‌تست طبق مقالاتِ معتبر و نقدِ روش‌شناسی
---

## چرا این بخش لازم شد؟

با بررسیِ ادبیاتِ معتبر (Whaley 2002 روی BXM؛ Hill, Balasubramanian, Gregory
& Tierens 2006، *FAJ*؛ Foltice 2022؛ Israelov & Nielsen 2014، AQR؛ Diaz &
Kwon 2019؛ و یک مطالعه‌ی ۲۰۲۳-۲۰۲۵ رویِ بازارهایِ نوظهور) و یک بازبینیِ کدِ
مستقل، چند مشکل در بک‌تستِ بخشِ ۴ شناسایی شد:

1. **فرکانسِ رول نااستاندارد** — رولِ ۱۰-۲۰ روزه در برابرِ استانداردِ ماهانه.
2. **بدونِ بنچمارکِ مکانیکی/غیرفعال**.
3. **بدونِ هزینه‌یِ معاملاتی** (و بعداً: فقط رویِ پرمیوم، نه رویِ خودِ سهم هم).
4. **بدونِ تفکیکِ زیر-دوره**.
5. **بدونِ بازه‌یِ اطمینان**.
6. **ناسازگاریِ روزِ معاملاتی/تقویمی** — `H`/`ROLL_DAYS` روزِ معاملاتی‌اند
   اما با `DAYCOUNT=365` (تقویمی) سالانه می‌شدند؛ باعثِ اشتباهِ سیستماتیک در
   پرمیوم، احتمالِ اعمال، و Sharpe می‌شد.
7. **نشتِ اطلاعاتِ آینده در وزنِ Black-Litterman و نسبتِ پوشش** — این دو، یک‌بار
   با کلِ دورهٔ Test محاسبه و در همه‌ی تاریخ‌هایِ بک‌تست (حتی سال‌هایِ اول)
   بازاستفاده می‌شدند؛ یعنی تصمیمِ سالِ ۲۰۲۲ از رکوردِ عملکردِ مدل تا ۲۰۲۵ خبر
   داشت.

این بخش هر هفت مورد را اصلاح می‌کند: ابتدا رولِ ماهانه + هزینه + بازه‌ی
اطمینان + تفکیکِ سالانه (نسخه‌یِ قبلی)، سپس رفعِ ایرادِ روزِ تقویمی، و در
انتها یک نسخه‌یِ کاملاً **رولینگ** که وزنِ پرتفو و نسبتِ پوشش را در **هر**
تاریخِ rebalance، فقط با داده‌یِ تا همان تاریخ، از نو می‌سازد.
"""))

new_cells.append(code(r"""
from scipy.stats import norm as _norm2

TRADING_DAYS_PER_YEAR = 252.0   # H و ROLL_DAYS از رویِ ایندکسِ روزِ معاملاتی‌اند
                                 # (نه تقویمی)؛ پس سالانه‌سازی/T باید با ۲۵۲ انجام
                                 # شود، نه با DAYCOUNT=365 (که مخصوصِ کمیت‌هایِ
                                 # واقعاً تقویمی مثلِ MATURITY_GRID در بخشِ ۳ است).
                                 # رفعِ ایرادِ «ناسازگاریِ روزِ معاملاتی/تقویمی».
ROLL_DAYS = 21           # قراردادِ استانداردِ ماهانه (Whaley 2002 BXM/BXY)؛
                          # با ۲۵۲ روزِ معاملاتی در سال، ۲۱ روز = دقیقاً ۱/۱۲ سال.
MECH_OTM = 0.02           # قاعده‌یِ ثابتِ BXY: همیشه ۲٪ خارج از پول
TXN_COST_PCT = 0.03       # فرضِ هزینه‌یِ معاملاتیِ رویِ پرمیومِ اختیار (اسپرد+کارمزد)
STOCK_TXN_COST_PCT = 0.01 # فرضِ هزینه‌یِ معاملاتیِ رویِ خودِ سهم (اسپرد+کارمزدِ
                          # خرید/فروش) — جدا از هزینه‌یِ پرمیوم؛ هر دو داده‌یِ
                          # واقعیِ bid-ask بازارِ ایران در دسترس نبود، پس این‌ها
                          # فرضِ صریح و قابلِ‌تنظیم‌اند، نه عددِ اندازه‌گیری‌شده.
                          # برایِ سادگی، این هزینه هر دوره کسر می‌شود (فرضِ
                          # محافظه‌کارانه‌تر از واقعیت، نه خوش‌بینانه‌تر).
N_BOOTSTRAP = 2000
np.random.seed(42)

# دیدگاهِ سالانه‌شده (mu_ann, sigma_ann) از همان پیش‌بینی‌هایِ خارج-از-نمونه‌یِ
# بخشِ ۱ — نسخه‌یِ *ثابت* (میانگینِ کلِ دوره)، برایِ دو نسخه‌یِ اولِ این بخش.
# نسخه‌یِ رولینگ (بدونِ نشتی) پایین‌تر جداگانه ساخته می‌شود.
views = {}
for name in ASSET_NAMES:
    H_orig = int(rec.loc[name, 'Horizon_days'])
    dfp = pd.read_csv(DATA_DIR + f'price_at_maturity_predictions_{name}.csv', parse_dates=['date']).set_index('date')
    S_t = processed_bt[name]['close'].reindex(dfp.index)
    z90 = _norm2.ppf(0.9)
    mu_ann_s = np.log(dfp['pred_ensemble'] / S_t) * TRADING_DAYS_PER_YEAR / H_orig
    sigma_ann_s = np.log(dfp['q90'].clip(lower=1.0) / dfp['q10'].clip(lower=1.0)) / (2 * z90 * np.sqrt(H_orig / TRADING_DAYS_PER_YEAR))
    sigma_ann_s = sigma_ann_s.clip(lower=0.05)
    views[name] = pd.DataFrame({'mu_ann': mu_ann_s, 'sigma_ann': sigma_ann_s}).dropna()

print("دیدگاهِ سالانه‌شده برایِ هر ۶ سهم آماده شد (بازاستفاده از پیش‌بینی‌هایِ بخشِ ۱).")
"""))

new_cells.append(code(r"""
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
        T = ROLL_DAYS / TRADING_DAYS_PER_YEAR   # رفعِ ایرادِ روزِ تقویمی: بود ROLL_DAYS/365

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
                ann_ret = (e_min / cost_basis_try) ** (TRADING_DAYS_PER_YEAR / ROLL_DAYS) - 1
                ann_var = (var_min / cost_basis_try ** 2) * (TRADING_DAYS_PER_YEAR / ROLL_DAYS)
                util = ann_ret - 0.5 * DELTA * ann_var
                if util > best_util:
                    best_util, K, premium = util, K_try, premium_try

        net_premium = premium * (1 - TXN_COST_PCT) if txn_cost else premium
        cost_basis = S0 - net_premium
        realized_min = min(actual_price, K)
        cc_covered_ret = (realized_min + net_premium) / cost_basis - 1
        uncovered_ret = actual_price / S0 - 1
        cc_ret = coverage * cc_covered_ret + (1 - coverage) * uncovered_ret
        if txn_cost:
            cc_ret -= STOCK_TXN_COST_PCT
            uncovered_ret_net = uncovered_ret - STOCK_TXN_COST_PCT
        else:
            uncovered_ret_net = uncovered_ret

        # پیش‌بینیِ بازده در همان لحظه‌ی انتخاب (نه بعد از وقوع) — طبقِ همان دیدگاهِ
        # فیزیکیِ (mu_ann, sigma_ann) که Strike را انتخاب کرد، برایِ مقایسه‌یِ بعدیِ
        # «پیش‌بینی‌شده در برابرِ واقعی».
        e_min_final, _, _ = covered_call_physical_moments(S0, K, T, mu_ann, sigma_ann)
        expected_covered_ret = e_min_final / cost_basis - 1
        expected_uncovered_ret = np.exp(mu_ann * T) - 1
        expected_ret = coverage * expected_covered_ret + (1 - coverage) * expected_uncovered_ret
        if txn_cost:
            expected_ret -= STOCK_TXN_COST_PCT

        rows.append(dict(date=t, S0=S0, K=K, premium=premium, actual_price=actual_price,
                          cc_ret=cc_ret, bh_ret=uncovered_ret_net, expected_ret=expected_ret, year=t.year))
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


def bootstrap_ci(returns, periods_per_year, n_boot=N_BOOTSTRAP, block_len=None):
    # Stationary bootstrap (Politis & Romano, 1994) به‌جایِ i.i.d. resampling —
    # بازده‌هایِ رول‌شده‌ی کاوردکال می‌توانند autocorrelation/persistence رژیمی
    # داشته باشند؛ i.i.d. این ساختار را نادیده می‌گیرد و بازه‌یِ اطمینان را
    # کاذب باریک می‌کند. طولِ بلوکِ میانگین با قاعده‌یِ رایجِ n^(1/3) انتخاب
    # می‌شود (Hall, Horowitz & Jing 1995) — یک heuristic استاندارد، نه
    # بهینه‌سازیِ رسمیِ ACF.
    returns = np.asarray(returns)
    n = len(returns)
    if n < 3:
        return dict(TotalReturn_lo=np.nan, TotalReturn_hi=np.nan, Sharpe_lo=np.nan, Sharpe_hi=np.nan)
    if block_len is None:
        block_len = max(2, int(round(n ** (1 / 3))))
    p_restart = 1.0 / block_len
    trs, shs = [], []
    for _ in range(n_boot):
        s = np.empty(n)
        i = np.random.randint(0, n)
        for k in range(n):
            s[k] = returns[i]
            i = np.random.randint(0, n) if np.random.rand() < p_restart else (i + 1) % n
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


periods_per_year_corrected = TRADING_DAYS_PER_YEAR / ROLL_DAYS
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
print("✅ بک‌تستِ اصلاح‌شده (روزِ تقویمیِ درست + هزینه‌یِ سهم و اختیار) اجرا شد.")
print(corrected_summary_df.round(3).to_string(index=False))
"""))

new_cells.append(md(r"""
---
## بخشِ ۴ب-۲ — نسخه‌یِ کاملاً رولینگ: بدونِ نشتِ اطلاعاتِ آینده

مهم‌ترین ایرادِ باقی‌مانده این بود که وزنِ Black-Litterman و نسبتِ پوشش، یک‌بار
با **کلِ** دورهٔ Test ساخته می‌شدند و در همه‌ی تاریخ‌هایِ بک‌تست (حتی سال‌هایِ
اول) بازاستفاده می‌شدند. این بخش دقیقاً همان محاسباتِ بخش‌هایِ ۲ و ۳ را،
اما **در هر تاریخِ rebalance جداگانه** و فقط با داده‌یِ تا همان تاریخ،
بازسازی می‌کند:

- کوواریانس: Ledoit-Wolf روی ۵۰۴ روزِ معاملاتیِ *گذشته* (تا t).
- Prior تعادلی: از وزنِ ارزشِ معاملاتیِ *گذشته* (تا t).
- دیدگاه (Q) و عدمِ‌قطعیتِ آن (Ω): از آخرین پیش‌بینیِ در دسترس در t، و از
  RMSE ای که فقط با پیش‌بینی‌هایِ **از پیش سررسیدشده** (تا t) محاسبه شده —
  نه میانگینِ کلِ آینده‌یِ Test.
- نسبتِ پوشش: از همان دیدگاهِ رولینگ، نه میانگینِ کلِ دوره.
"""))

new_cells.append(code(r"""
from sklearn.covariance import LedoitWolf as _LedoitWolf2
from scipy.optimize import minimize as _minimize2

BL_COV_WINDOW = 504
BL_TAU = 0.05
BL_W_MIN, BL_W_MAX = 0.05, 0.30
MIN_MATURED_FOR_RMSE = 5   # حداقل تعدادِ پیش‌بینیِ سررسیدشده برایِ محاسبه‌یِ RMSEِ رولینگ

asset_pred_detail = {}
for name in ASSET_NAMES:
    H_orig = int(rec.loc[name, 'Horizon_days'])
    dfp = pd.read_csv(DATA_DIR + f'price_at_maturity_predictions_{name}.csv', parse_dates=['date']).set_index('date')
    close_s = processed_bt[name]['close']
    S_t = close_s.reindex(dfp.index)
    z90 = _norm2.ppf(0.9)
    mu_ann_i = np.log(dfp['pred_ensemble'] / S_t) * TRADING_DAYS_PER_YEAR / H_orig
    sigma_ann_i = np.log(dfp['q90'].clip(lower=1.0) / dfp['q10'].clip(lower=1.0)) / (2 * z90 * np.sqrt(H_orig / TRADING_DAYS_PER_YEAR))
    sigma_ann_i = sigma_ann_i.clip(lower=0.05)
    logret_pred_i = np.log(dfp['pred_ensemble'] / S_t)
    logret_actual_i = np.log(dfp['actual_price'] / S_t)
    sq_err_i = (logret_pred_i - logret_actual_i) ** 2

    mat_dates = []
    for d in dfp.index:
        if d not in close_s.index:
            mat_dates.append(pd.NaT); continue
        pos = close_s.index.get_loc(d)
        mat_dates.append(close_s.index[pos + H_orig] if pos + H_orig < len(close_s) else pd.NaT)

    detail = pd.DataFrame({'mu_ann': mu_ann_i, 'sigma_ann': sigma_ann_i, 'sq_err': sq_err_i,
                            'maturity_date': mat_dates}, index=dfp.index).dropna()
    detail['H'] = H_orig
    asset_pred_detail[name] = detail

print("جزئیاتِ رولینگِ پیش‌بینی (دیدگاه + خطایِ سررسیدشده) برایِ هر ۶ سهم آماده شد.")


def rolling_view_at(name, t):
    d = asset_pred_detail[name]
    avail = d.loc[:t]
    if len(avail) == 0:
        return None
    latest = avail.iloc[-1]
    matured = d[d['maturity_date'] <= t]
    if len(matured) >= MIN_MATURED_FOR_RMSE:
        rmse_logret_t = np.sqrt(matured['sq_err'].mean())
    else:
        rmse_logret_t = np.sqrt(avail['sq_err'].mean())  # هنوز دیتای سررسیدشده‌ی کافی نیست؛ fallback محافظه‌کارانه
    return dict(mu_ann=latest['mu_ann'], sigma_ann=latest['sigma_ann'], rmse_logret=rmse_logret_t, H=int(latest['H']))


ret_wide_bt = pd.DataFrame({name: np.log(processed_bt[name]['close'] / processed_bt[name]['close'].shift(1))
                             for name in ASSET_NAMES}).dropna()


def rolling_bl_at(t):
    hist = ret_wide_bt.loc[:t].iloc[-BL_COV_WINDOW:]
    if len(hist) < 60:
        return None
    lw = _LedoitWolf2().fit(hist.values)
    Sigma_t = lw.covariance_ * 252

    value_hist = pd.Series({name: processed_bt[name]['value'].reindex(hist.index).mean() for name in ASSET_NAMES})
    w_mkt_t = (value_hist / value_hist.sum()).reindex(ASSET_NAMES).values
    Pi_t = DELTA * Sigma_t @ w_mkt_t

    views_t = {name: rolling_view_at(name, t) for name in ASSET_NAMES}
    if any(v is None for v in views_t.values()):
        return None

    Q_t = np.array([views_t[name]['mu_ann'] for name in ASSET_NAMES])
    omega_diag_t = np.array([(views_t[name]['rmse_logret'] ** 2) * (TRADING_DAYS_PER_YEAR / views_t[name]['H'])
                              for name in ASSET_NAMES])
    Omega_t = np.diag(omega_diag_t)
    P = np.eye(len(ASSET_NAMES))
    tauSigma_inv = np.linalg.inv(BL_TAU * Sigma_t)
    Omega_inv = np.linalg.inv(Omega_t)
    M = np.linalg.inv(tauSigma_inv + P.T @ Omega_inv @ P)
    Pi_BL_t = M @ (tauSigma_inv @ Pi_t + P.T @ Omega_inv @ Q_t)

    n = len(ASSET_NAMES)
    def neg_sharpe_t(w):
        ret = w @ Pi_BL_t
        vol = np.sqrt(w @ Sigma_t @ w)
        return -(ret - RISK_FREE_RATE) / vol
    bounds = [(BL_W_MIN, BL_W_MAX)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    res = _minimize2(neg_sharpe_t, np.ones(n) / n, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        return None
    w_t = pd.Series(res.x / res.x.sum(), index=ASSET_NAMES)

    sigma_views_t = pd.Series({name: views_t[name]['sigma_ann'] for name in ASSET_NAMES})
    u_norm = (sigma_views_t - sigma_views_t.min()) / (sigma_views_t.max() - sigma_views_t.min() + 1e-12)
    coverage_t = 0.90 - 0.40 * u_norm

    return dict(weights=w_t, coverage=coverage_t, views=views_t)


common_start = max(asset_pred_detail[name].index.min() for name in ASSET_NAMES)
common_end = min(asset_pred_detail[name].index.max() for name in ASSET_NAMES)
common_calendar = processed_bt[ASSET_NAMES[0]]['close'].index
for name in ASSET_NAMES[1:]:
    common_calendar = common_calendar.intersection(processed_bt[name]['close'].index)
common_calendar = common_calendar[(common_calendar >= common_start) & (common_calendar <= common_end)]
reb_dates_common = common_calendar[::ROLL_DAYS]

rolling_rows = {name: [] for name in ASSET_NAMES}
rolling_weight_history = []
skipped_dates = 0

for t in reb_dates_common:
    bl = rolling_bl_at(t)
    if bl is None:
        skipped_dates += 1
        continue
    weights_t, coverage_t, views_t = bl['weights'], bl['coverage'], bl['views']
    rolling_weight_history.append(dict(date=t, **weights_t.to_dict()))

    for name in ASSET_NAMES:
        close = processed_bt[name]['close']
        if t not in close.index:
            continue
        pos = close.index.get_loc(t)
        if pos + ROLL_DAYS >= len(close):
            continue
        S0 = close.iloc[pos]
        actual_price = close.iloc[pos + ROLL_DAYS]
        mu_ann, sigma_ann = views_t[name]['mu_ann'], views_t[name]['sigma_ann']

        hist = close.loc[:t].pct_change().dropna().iloc[-60:]
        sigma_bs = hist.std() * np.sqrt(252) if len(hist) > 10 else sigma_ann
        T = ROLL_DAYS / TRADING_DAYS_PER_YEAR

        best_util, K, premium = -np.inf, None, None
        for otm in OTM_GRID:
            K_try = S0 * (1 + otm)
            premium_try = black_scholes_call(S0, K_try, T, RISK_FREE_RATE, sigma_bs)
            e_min, var_min, _ = covered_call_physical_moments(S0, K_try, T, mu_ann, sigma_ann)
            cost_basis_try = S0 - premium_try
            ann_ret = (e_min / cost_basis_try) ** (TRADING_DAYS_PER_YEAR / ROLL_DAYS) - 1
            ann_var = (var_min / cost_basis_try ** 2) * (TRADING_DAYS_PER_YEAR / ROLL_DAYS)
            util = ann_ret - 0.5 * DELTA * ann_var
            if util > best_util:
                best_util, K, premium = util, K_try, premium_try

        net_premium = premium * (1 - TXN_COST_PCT)
        cost_basis = S0 - net_premium
        realized_min = min(actual_price, K)
        cc_covered_ret = (realized_min + net_premium) / cost_basis - 1
        uncovered_ret = actual_price / S0 - 1
        coverage = float(coverage_t[name])
        cc_ret = (coverage * cc_covered_ret + (1 - coverage) * uncovered_ret) - STOCK_TXN_COST_PCT
        bh_ret = uncovered_ret - STOCK_TXN_COST_PCT

        # پیش‌بینیِ بازده در همان لحظه‌ی انتخاب (نه بعد از وقوع) — طبقِ همان دیدگاهِ
        # (mu_ann, sigma_ann) که Strike را انتخاب کرد، برایِ مقایسه‌یِ بعدیِ
        # «پیش‌بینی‌شده در برابرِ واقعی».
        e_min_final, _, _ = covered_call_physical_moments(S0, K, T, mu_ann, sigma_ann)
        expected_covered_ret = e_min_final / cost_basis - 1
        expected_uncovered_ret = np.exp(mu_ann * T) - 1
        expected_ret = (coverage * expected_covered_ret + (1 - coverage) * expected_uncovered_ret) - STOCK_TXN_COST_PCT

        rolling_rows[name].append(dict(date=t, S0=S0, K=K, premium=premium, actual_price=actual_price,
                                        weight=float(weights_t[name]), coverage=coverage,
                                        cc_ret=cc_ret, bh_ret=bh_ret, expected_ret=expected_ret, year=t.year))

rolling_results = {name: pd.DataFrame(rows).set_index('date') for name, rows in rolling_rows.items()}
weight_history_df = pd.DataFrame(rolling_weight_history).set_index('date')
print(f"✅ بک‌تستِ رولینگِ Black-Litterman اجرا شد — {len(reb_dates_common)} تاریخِ rebalance "
      f"({skipped_dates} رد شد به دلیلِ کمبودِ داده‌یِ تاریخچه).")
print("\nمیانگین و بازه‌یِ وزنِ هر سهم در طولِ بک‌تست (رولینگ، نه ثابت):")
print(weight_history_df.describe().T[['mean', 'min', 'max']].round(3))
"""))

new_cells.append(md(r"""
## سطحِ پرتفو: مقایسه‌یِ سه نسخه

مکانیکی (قاعده‌یِ ثابتِ BXY)، بهینه‌شده (وزنِ ثابت اما با هزینه و روزِ
تقویمیِ درست)، و رولینگِ کامل (بدونِ نشتِ اطلاعاتِ آینده) — نسخه‌یِ اولیه‌ی
معیوب (بدونِ رولِ ماهانه، بدونِ هزینه، با نشتِ اطلاعات) در همین بازبینی حذف
شد و هرگز به این جدول راه نیافت.
"""))

new_cells.append(code(r"""
def portfolio_agg(results_dict, variant_name, periods_per_year):
    date_union = sorted(set().union(*[set(results_dict[variant_name][n].index) for n in ASSET_NAMES]))
    port_cc, port_bh, port_exp = [], [], []
    for t in date_union:
        cc_t, bh_t, exp_t, w_t = 0.0, 0.0, 0.0, 0.0
        for name in ASSET_NAMES:
            df_bt = results_dict[variant_name][name]
            if t in df_bt.index:
                w = weights_df.loc[name, 'Weight']
                cc_t += w * df_bt.loc[t, 'cc_ret']
                bh_t += w * df_bt.loc[t, 'bh_ret']
                exp_t += w * df_bt.loc[t, 'expected_ret']
                w_t += w
        if w_t > 0:
            port_cc.append(cc_t / w_t)
            port_bh.append(bh_t / w_t)
            port_exp.append(exp_t / w_t)
    return np.array(port_cc), np.array(port_bh), np.array(port_exp)


def portfolio_agg_rolling(results_dict):
    date_union = sorted(set().union(*[set(results_dict[n].index) for n in ASSET_NAMES]))
    port_cc, port_bh, port_exp = [], [], []
    for t in date_union:
        cc_t, bh_t, exp_t, w_t = 0.0, 0.0, 0.0, 0.0
        for name in ASSET_NAMES:
            df_bt = results_dict[name]
            if t in df_bt.index:
                w = df_bt.loc[t, 'weight']
                cc_t += w * df_bt.loc[t, 'cc_ret']
                bh_t += w * df_bt.loc[t, 'bh_ret']
                exp_t += w * df_bt.loc[t, 'expected_ret']
                w_t += w
        if w_t > 0:
            port_cc.append(cc_t / w_t)
            port_bh.append(bh_t / w_t)
            port_exp.append(exp_t / w_t)
    return np.array(port_cc), np.array(port_bh), np.array(port_exp)


portfolio_compare_rows = []

predicted_vs_actual_rows = []
expected_series_by_variant = {}

for variant_name in ['Mechanical_Monthly', 'Optimized_Monthly']:
    pcc, pbh, pexp = portfolio_agg(corrected_results, variant_name, periods_per_year_corrected)
    p = perf_metrics2(pcc, periods_per_year_corrected)
    ci = bootstrap_ci(pcc, periods_per_year_corrected)
    portfolio_compare_rows.append(dict(Variant=variant_name, TotalReturn=p['TotalReturn'], Sharpe=p['Sharpe'],
                                        TR_CI90_lo=ci['TotalReturn_lo'], TR_CI90_hi=ci['TotalReturn_hi']))
    expected_series_by_variant[variant_name] = (pexp, pcc)

pcc_roll, pbh_roll, pexp_roll = portfolio_agg_rolling(rolling_results)
p_roll = perf_metrics2(pcc_roll, periods_per_year_corrected)
ci_roll = bootstrap_ci(pcc_roll, periods_per_year_corrected)
portfolio_compare_rows.append(dict(Variant='Rolling_BL_Monthly (بدونِ نشتِ اطلاعاتِ آینده)',
                                    TotalReturn=p_roll['TotalReturn'], Sharpe=p_roll['Sharpe'],
                                    TR_CI90_lo=ci_roll['TotalReturn_lo'], TR_CI90_hi=ci_roll['TotalReturn_hi']))
expected_series_by_variant['Rolling_BL_Monthly'] = (pexp_roll, pcc_roll)

portfolio_compare_df = pd.DataFrame(portfolio_compare_rows)
print(portfolio_compare_df.round(3).to_string(index=False))
"""))

new_cells.append(md(r"""
## پیش‌بینی‌شده در برابرِ واقعی — سطحِ دوره و سطحِ پرتفو

تا این‌جا فقط بازدهِ **واقعی/رخ‌داده** (`cc_ret`) در بک‌تست ذخیره و گزارش
می‌شد؛ بازدهی که در **همان لحظه‌ی انتخابِ Strike** با همان دیدگاهِ فیزیکیِ
(`mu_ann`, `sigma_ann`) پیش‌بینی شده بود، محاسبه می‌شد اما هیچ‌جا ذخیره
نمی‌شد. اینجا آن پیش‌بینی (`expected_ret`) هم برای هر دوره نگه داشته شده و
حالا با بازدهِ واقعیِ همان دوره مقایسه می‌شود — هم به‌ازایِ هر دوره (ضریبِ
هم‌بستگی + نرخِ هم‌جهتی/Hit Rate)، هم در سطحِ کلِ پرتفو (بازدهِ کلیِ
پیش‌بینی‌شده به‌روشِ ترکیبی در برابرِ بازدهِ کلیِ واقعاً رخ‌داده).
"""))

new_cells.append(code(r"""
pred_vs_actual_rows = []
for variant_name, (pexp, pcc) in expected_series_by_variant.items():
    p_pred = perf_metrics2(pexp, periods_per_year_corrected)
    p_actual = perf_metrics2(pcc, periods_per_year_corrected)
    hit_rate = float((np.sign(pexp) == np.sign(pcc)).mean())
    corr = float(np.corrcoef(pexp, pcc)[0, 1]) if len(pexp) > 1 else np.nan
    pred_vs_actual_rows.append(dict(
        Variant=variant_name, N_Periods=len(pcc),
        Predicted_TotalReturn=p_pred['TotalReturn'], Predicted_Sharpe=p_pred['Sharpe'],
        Actual_TotalReturn=p_actual['TotalReturn'], Actual_Sharpe=p_actual['Sharpe'],
        HitRate_SameSign=hit_rate, Corr_Predicted_vs_Actual=corr,
    ))
predicted_vs_actual_df = pd.DataFrame(pred_vs_actual_rows)
print(predicted_vs_actual_df.round(3).to_string(index=False))
print("\nHitRate_SameSign: چند درصدِ دوره‌ها، پیش‌بینی و واقعیت هم‌جهت بودند (هر دو سود یا هر دو ضرر).")
print("Corr_Predicted_vs_Actual: هم‌بستگیِ اندازه‌یِ بازدهِ پیش‌بینی‌شده با بازدهِ واقعی، دوره‌به‌دوره.")
"""))

new_cells.append(md(r"## نمودار: بازدهِ تجمعیِ پیش‌بینی‌شده در برابرِ واقعیِ Rolling_BL_Monthly"))

new_cells.append(code(r"""
pexp_r, pcc_r = expected_series_by_variant['Rolling_BL_Monthly']
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.cumprod(1 + pexp_r), label='پیش‌بینی‌شده (Expected)', lw=1.6)
ax.plot(np.cumprod(1 + pcc_r), label='واقعی (Actual)', lw=1.6)
ax.set_title('Rolling_BL_Monthly — رشدِ تجمعیِ سرمایه: پیش‌بینی‌شده در برابرِ واقعی')
ax.set_ylabel('رشدِ تجمعی (۱ = سرمایه‌ی اولیه)')
ax.legend()
plt.tight_layout()
plt.show()

print(f"اختلافِ بازدهِ کلِ پیش‌بینی‌شده و واقعی برایِ Rolling_BL_Monthly: "
      f"{(np.cumprod(1 + pexp_r)[-1] - np.cumprod(1 + pcc_r)[-1]):+.3f} "
      f"(به‌صورتِ ضریبِ رشدِ سرمایه)")
"""))

new_cells.append(md(r"""
## حساسیتِ Sharpeِ پرتفو به نرخِ بدونِ ریسک

نرخِ بدونِ ریسکِ ایران در این چند سال ثابت نبوده؛ ما در کلِ پروژه ۲۰٪ فرض
کرده‌ایم. این جدول نشان می‌دهد اگر آن فرض جابه‌جا شود، Sharpeِ نسخه‌یِ
رولینگ چطور تغییر می‌کند (⚠️ فقط معیارِ Sharpe عوض می‌شود؛ قیمت‌گذاریِ
اختیار خودش هنوز با rf=۲۰٪ محاسبه شده — یک بازسازیِ کاملِ حساسیت باید کلِ
انتخابِ Strike را هم برایِ هر rf از نو اجرا کند که در این نسخه انجام
نشده است).
"""))

new_cells.append(code(r"""
years_roll = len(pcc_roll) / periods_per_year_corrected
equity_roll = np.cumprod(1 + pcc_roll)
cagr_roll = equity_roll[-1] ** (1 / years_roll) - 1
vol_roll = pcc_roll.std() * np.sqrt(periods_per_year_corrected)

rf_sensitivity_rows = []
for rf_test in [0.10, 0.20, 0.30, 0.40]:
    sharpe_rf = (cagr_roll - rf_test) / vol_roll
    rf_sensitivity_rows.append(dict(Risk_Free_Rate=rf_test, Portfolio_CAGR=cagr_roll,
                                     Portfolio_Vol_Ann=vol_roll, Sharpe=sharpe_rf))
rf_sensitivity_df = pd.DataFrame(rf_sensitivity_rows)
print(rf_sensitivity_df.round(3).to_string(index=False))
"""))

new_cells.append(md(r"""
## تفکیکِ زیر-دوره (به‌سال) — نسخه‌یِ بهینه‌شده‌یِ ماهانه (وزنِ ثابت)

سال‌هایی با کمتر از ۳ دوره حذف شده‌اند (Sharpe سالانه‌شده با نمونه‌ی خیلی
کوچک بی‌معنی و انفجاری می‌شود).
"""))

new_cells.append(code(r"""
sp = subperiod_df[subperiod_df.Variant == 'Optimized_Monthly'].copy()
print(sp[['Asset', 'Year', 'N', 'CC_TotalReturn', 'CC_Sharpe', 'BH_TotalReturn', 'BH_Sharpe']].round(3).to_string(index=False))

print("\nمیانگینِ Sharpe در هر سال (روی همه‌ی سهم‌ها):")
print(sp.groupby('Year')[['CC_Sharpe', 'BH_Sharpe']].mean().round(3))
"""))

new_cells.append(md(r"""
## جمع‌بندیِ صادقانه‌یِ بازبینی

**آنچه در این نسخه اصلاح شد:**
- رولِ ماهانه + هزینه‌یِ معاملاتیِ سهم و اختیار + بازه‌یِ اطمینان + تفکیکِ سالانه
  (از بازبینیِ اول).
- **روزِ معاملاتی/تقویمی درست شد**: `H`/`ROLL_DAYS` حالا با ۲۵۲ (نه ۳۶۵)
  سالانه می‌شوند — پرمیوم، احتمالِ اعمال، و Sharpe همه تغییر کردند.
- **وزنِ Black-Litterman و نسبتِ پوشش دیگر ثابت نیستند** — نسخه‌یِ
  `Rolling_BL_Monthly` این‌ها را در **هر** تاریخِ rebalance، فقط با داده‌یِ
  تا همان تاریخ (کوواریانس، Prior، دیدگاه، RMSEِ رولینگ)، از نو می‌سازد. این
  مستقیماً نشتِ اطلاعاتِ آینده به وزن/coverage را حذف می‌کند.
- هزینه‌یِ معاملاتی حالا هم رویِ پرمیومِ اختیار و هم رویِ خودِ سهم اعمال می‌شود.

**نتیجه:** رفعِ همزمانِ همه‌ی این ایرادها، نسبت به نسخه‌ی اولیه‌ی معیوب (که
هرگز به گزارشِ نهایی راه نیافت)، عددِ `Rolling_BL_Monthly` را به سمتِ مقادیرِ
به‌مراتب متواضعانه‌تر و قابلِ‌دفاع‌تر می‌برد.

**لایه‌یِ دو هم در این نسخه اضافه شد:**
- بازه‌یِ اطمینانِ Bootstrap دیگر i.i.d. نیست — از **Stationary Bootstrap**
  (Politis & Romano, 1994) با طولِ بلوکِ `n^(1/3)` استفاده می‌کند تا
  autocorrelation/persistence بینِ دوره‌هایِ رول‌شده را نادیده نگیرد.
- نشتِ مرزیِ انتخابِ افق (بخشِ ۱، تورنمنتِ Horizon)، پنجره‌یِ fit‌شدنِ GARCH
  (که قبلاً Train+Val را می‌دید، نه فقط Train)، نشتِ StandardScaler درونِ
  GridSearchCVِ Ridge، و quantile crossing در مدل‌هایِ کوانتایل — همه در
  `price_at_maturity_prediction.ipynb` اصلاح و کلِ Pipeline از نو اجرا شد.

**آنچه هنوز اصلاح نشده (فراتر از دامنه‌یِ این بازبینی، عمدتاً به‌خاطرِ نبودِ
داده):** قیمتِ واقعیِ بازارِ آپشن (به‌جایِ بلک-شولزِ نظری)، Volatility Risk
Premium به‌صورتِ مستقیم (چون implied volatilityِ واقعی نداریم)، و NAVِ
روزانه‌یِ کاملِ نقد/تسویه. کنترلِ multiple-testing (Deflated Sharpe Ratio)،
حساسیت به دامنه‌ی نوسانِ روزانه، و حساسیتِ آستانه‌یِ رویدادِ شرکتی — هر سه در
بخشِ ۴ب-۳ (پایین‌تر) اضافه شدند.

**منابع:** Whaley (2002); Feldman & Roy (2005); Hill, Balasubramanian,
Gregory & Tierens (2006, *FAJ*); Foltice (2022); Israelov & Nielsen (2014,
AQR); Israelov & Klein (2016); Diaz & Kwon (2019, *Journal of Asset
Management*); مطالعه‌یِ بازارهایِ نوظهور (۲۰۲۱-۲۰۲۵).
"""))


new_cells.append(md(r"""
---
## بخشِ ۴ب-۳ — لایه‌یِ سه: کنترلِ Multiple-Testing، دامنه‌یِ نوسان، و حساسیتِ آستانه‌یِ رویدادِ شرکتی
---

### ۱) Deflated Sharpe Ratio — آیا Sharpe=۰.۷۵ می‌تواند صرفاً محصولِ جست‌وجویِ زیاد باشد؟

در طولِ این پروژه ده‌ها انتخاب آزاد انجام شد (۴ کاندیدایِ افق × ۹ مدل ×
شبکه‌یِ ۱۱تاییِ Strike × چند نسخه‌یِ متفاوتِ بک‌تست). طبقِ Bailey & López de
Prado (2014), "The Deflated Sharpe Ratio"، وقتی از میانِ N آزمایش بهترین
Sharpe گزارش می‌شود، حتی اگر هیچ مهارتِ واقعی‌ای در کار نباشد، بهترینِ N
آزمایشِ تصادفی هم به‌طورِ سیستماتیک Sharpeِ مثبت نشان می‌دهد (اریبیِ انتخاب).
DSR این اریبی را تصحیح می‌کند و می‌پرسد: «با احتسابِ N آزمایش، احتمالِ اینکه
مهارتِ واقعی پشتِ این Sharpe باشد چقدر است؟» چون شمارشِ دقیقِ N بحث‌برانگیز
است، آن را در یک بازه (۵ تا ۲۰۰) حساسیت‌سنجی می‌کنیم، نه یک عددِ واحد.
"""))

new_cells.append(code(r"""
from scipy.stats import skew as _skew, kurtosis as _kurtosis, norm as _norm3

def deflated_sharpe_ratio(returns, periods_per_year, n_trials, rf_ann=RISK_FREE_RATE):
    returns = np.asarray(returns)
    n = len(returns)
    rf_per_period = rf_ann / periods_per_year
    excess = returns - rf_per_period
    sr_hat = excess.mean() / excess.std()          # Sharpeِ دوره‌ای (غیرِ سالانه‌شده)
    g3 = _skew(excess)
    g4 = _kurtosis(excess, fisher=False)            # کشیدگیِ پیرسون (نرمال = ۳)
    euler_gamma = 0.5772156649015329
    var_sr_trials = 1.0 / max(n - 1, 1)              # برآوردِ واریانسِ Sharpeِ زیرِ فرضِ صفر (تقریبِ استاندارد)
    sr0 = np.sqrt(var_sr_trials) * ((1 - euler_gamma) * _norm3.ppf(1 - 1.0 / n_trials) +
                                     euler_gamma * _norm3.ppf(1 - 1.0 / (n_trials * np.e)))
    denom = np.sqrt(max(1 - g3 * sr_hat + ((g4 - 1) / 4) * sr_hat ** 2, 1e-8))
    psr = _norm3.cdf((sr_hat - sr0) * np.sqrt(max(n - 1, 1)) / denom)
    return dict(N=n, SR_hat_per_period=sr_hat, Skew=g3, Kurtosis=g4,
                SR0_expected_max_under_null=sr0, DSR=psr)


dsr_rows = []
for n_trials_test in [5, 20, 50, 100, 200]:
    r = deflated_sharpe_ratio(pcc_roll, periods_per_year_corrected, n_trials_test)
    dsr_rows.append(dict(N_Trials_Assumed=n_trials_test, DSR_Confidence=r['DSR'],
                          SR0_Hurdle=r['SR0_expected_max_under_null']))
dsr_df = pd.DataFrame(dsr_rows)
print("Deflated Sharpe Ratio برایِ پرتفویِ Rolling_BL_Monthly، به‌ازایِ چند فرضِ N (تعدادِ آزمایش):")
print(dsr_df.round(4).to_string(index=False))
print(f"\nSkewness دوره‌ای: {_skew(pcc_roll):.3f} | Kurtosis: {_kurtosis(pcc_roll, fisher=False):.3f}")
print("\nتفسیر: DSR نزدیکِ ۱ یعنی حتی با احتسابِ جست‌وجویِ زیاد، بعید است این Sharpe کاملاً شانسی باشد؛")
print("نزدیکِ ۰.۵ یا کمتر یعنی نمی‌توان مهارتِ واقعی را از نویزِ ناشیِ از جست‌وجویِ زیاد جدا کرد.")
"""))

new_cells.append(md(r"""
### ۲) دامنه‌یِ نوسانِ روزانه — آیا Strikeهایِ پیشنهادی اصلاً در بازه‌یِ نگهداری قابلِ‌دسترس‌اند؟

بورسِ تهران محدودیتِ دامنه‌یِ نوسانِ روزانه دارد (سهم نمی‌تواند در یک روز بیش
از حدِ مشخصی جهش کند). این یک محدودیتِ میکرواستراکچر است که بلک-شولزِ
استاندارد (بازارِ پیوسته و بدونِ اصطکاک) آن را نادیده می‌گیرد. عددِ دقیقِ
مجازِ فعلی برایِ هر نماد را در این محیط نمی‌توانیم راستی‌آزمایی کنیم؛ به‌جای
ادعای یک عددِ قطعی، حداکثرِ جابه‌جاییِ نظریِ ممکن را زیرِ **چند فرضِ محتمل**
(۳٪، ۵٪، ۷٪ در روز) حساب می‌کنیم و می‌بینیم Strikeِ پیشنهادی برایِ هر سهم زیرِ
کدام فرض‌ها اصلاً در بازه‌یِ ۲۱روزه قابلِ‌دسترس است.
"""))

new_cells.append(code(r"""
price_limit_rows = []
for name in ASSET_NAMES:
    otm_selected = final_rec.loc[name, 'OTM_pct']
    for daily_limit in [0.03, 0.05, 0.07]:
        max_cum_move = (1 + daily_limit) ** ROLL_DAYS - 1
        reachable = abs(otm_selected) <= max_cum_move
        price_limit_rows.append(dict(Asset=name, Selected_OTM_pct=otm_selected,
                                      Assumed_Daily_Limit=daily_limit,
                                      Max_Cumulative_Move_in_ROLL_DAYS=max_cum_move,
                                      Reachable=reachable))
price_limit_df = pd.DataFrame(price_limit_rows)
pivot = price_limit_df.pivot(index='Asset', columns='Assumed_Daily_Limit', values='Reachable')
print("آیا Strikeِ انتخاب‌شده زیرِ هر فرضِ دامنه‌ی نوسانِ روزانه در ۲۱ روز قابلِ‌دسترس است؟")
print(pivot)
print("\n⚠️ اعدادِ دامنه‌ی نوسان (۳٪/۵٪/۷٪) فرضی‌اند، نه عددِ رسمیِ راستی‌آزمایی‌شده‌ی TSE برایِ این نمادها؛")
print("   این جدول فقط حساسیت را نشان می‌دهد، نه یک اصلاحِ قطعی رویِ پرمیوم یا بک‌تست.")
"""))

new_cells.append(md(r"""
### ۳) حساسیتِ آستانه‌یِ تشخیصِ رویدادِ شرکتی (۲۵٪ ثابت)

آستانه‌ی ۲۵٪ (که در همه‌ی نوت‌بوک‌ها برایِ تشخیصِ جهش‌هایِ ناشی از رویدادِ
شرکتی استفاده شد) یک قاعده‌یِ ساده و بدونِ تقویمِ رسمی است. اینجا نشان
می‌دهیم تعدادِ رویدادهایِ تشخیص‌داده‌شده به این آستانه چقدر حساس است — اگر
تعداد با تغییرِ کوچکِ آستانه به‌شدت عوض شود، یعنی این قاعده شکننده است.
"""))

new_cells.append(code(r"""
threshold_sensitivity_rows = []
for name in ASSET_NAMES:
    df_raw = load_and_clean(name)
    close_raw = df_raw['close'].astype(float).values
    daily_chg = np.abs(np.diff(close_raw) / close_raw[:-1])
    for thr in [0.15, 0.20, 0.25, 0.30, 0.35]:
        n_events = int((daily_chg > thr).sum())
        threshold_sensitivity_rows.append(dict(Asset=name, Threshold=thr, N_Detected_Events=n_events))
threshold_sensitivity_df = pd.DataFrame(threshold_sensitivity_rows)
pivot2 = threshold_sensitivity_df.pivot(index='Asset', columns='Threshold', values='N_Detected_Events')
print("تعدادِ رویدادِ تشخیص‌داده‌شده به‌عنوانِ «رویدادِ شرکتی» به‌ازایِ آستانه‌هایِ مختلف:")
print(pivot2)
print(f"\nآستانه‌یِ فعلیِ پروژه: {CORP_ACTION_THRESHOLD:.0%}")
print("اگر تعداد بینِ ستون‌هایِ نزدیک به هم خیلی فرق کند، یعنی نتیجه به این پارامترِ")
print("دلخواه حساس است و باید در محدودیت‌هایِ پایان‌نامه صریح ذکر شود.")
"""))


new_cells.append(md(r"""
---
# بخشِ ۵ب — سه اصلاحِ اولویت‌دار طبقِ Diaz & Kwon (2020)
---

با مقایسه‌ی مقاله‌ی دومِ Diaz & Kwon («Optimization of covered calls under
uncertainty»، ۲۰۲۰) سه اختلافِ اولویت‌دار با کارِ ما شناسایی شد که همه بدونِ
نیاز به داده‌ی جدید قابلِ‌پیاده‌سازی بودند:

1. **ترکیبِ چند سررسیدِ همزمان** (اینجا: ۱۰ روزه + ۲۱ روزه، به‌جایِ فقط ۲۱ روزه).
2. **تابعِ مطلوبیتِ Quadratic** به‌جایِ CVaR (چارچوبِ عمومی‌ترِ مقاله‌ی ۲۰۲۰).
3. **هزینه‌ی معاملاتیِ مبتنی‌بر گردش (turnover)** به‌جایِ درصدِ ثابتِ ساده.

این بخش نسخه‌ی جدیدی از LP (اینجا QP، چون Quadratic Utility محدبیتِ درجه‌دومی
اضافه می‌کند) با کتابخانه‌ی `cvxpy` می‌سازد و آن را با همان تاریخ‌هایِ
rebalanceِ مشترک اجرا می‌کند.
"""))

new_cells.append(code(r"""
import cvxpy as cp

SHORT_MATURITY_DAYS = 10     # سررسیدِ کوتاه (روزِ معاملاتی)
S_UTIL = 2.5                 # پارامترِ Quadratic Utility (McMillan: ARA=1/(s-W)≈1/(s-1))
TURNOVER_COST = 0.01         # هزینه‌یِ هر واحدِ گردشِ پرتفو (c_z در نمادِ مقاله)
TURNOVER_MAX = 0.60          # سقفِ گردشِ کلِ پرتفو در هر rebalance


def simulate_two_horizon(S0_vec, mu_vec, sigma_vec, corr, T1, T2, n_scen, rng):
    n_a = len(S0_vec)
    L = np.linalg.cholesky(corr + 1e-8 * np.eye(n_a))
    Z1 = rng.standard_normal((n_scen, n_a)) @ L.T
    Z2 = rng.standard_normal((n_scen, n_a)) @ L.T
    drift1 = (mu_vec - 0.5 * sigma_vec ** 2) * T1
    S_T1 = S0_vec[None, :] * np.exp(drift1[None, :] + sigma_vec[None, :] * np.sqrt(T1) * Z1)
    drift2 = (mu_vec - 0.5 * sigma_vec ** 2) * (T2 - T1)
    S_T2 = S_T1 * np.exp(drift2[None, :] + sigma_vec[None, :] * np.sqrt(T2 - T1) * Z2)
    return S_T1, S_T2


def solve_multi_maturity_qp(S0_vec, mu_vec, sigma_vec, sigma_bs_vec, corr, T1, T2, otm_grid, rf,
                             w_min, w_max, s_util, turnover_cost, turnover_max, w_prev, n_scen, rng):
    n_a, n_k = len(S0_vec), len(otm_grid)
    S_T1, S_T2 = simulate_two_horizon(S0_vec, mu_vec, sigma_vec, corr, T1, T2, n_scen, rng)

    K_s = np.array([[S0_vec[j] * (1 + o) for o in otm_grid] for j in range(n_a)])
    K_l = K_s.copy()
    prem_s = np.array([[black_scholes_call(S0_vec[j], K_s[j, l], T1, rf, sigma_bs_vec[j])
                         for l in range(n_k)] for j in range(n_a)])
    prem_l = np.array([[black_scholes_call(S0_vec[j], K_l[j, l], T2, rf, sigma_bs_vec[j])
                         for l in range(n_k)] for j in range(n_a)])

    w = cp.Variable(n_a)
    p_s = cp.Variable((n_a, n_k), nonneg=True)
    p_l = cp.Variable((n_a, n_k), nonneg=True)
    x = cp.multiply(w, 1.0 / S0_vec)

    payout_s = np.maximum(S_T1[:, :, None] - K_s[None, :, :], 0.0)
    payout_l = np.maximum(S_T2[:, :, None] - K_l[None, :, :], 0.0)
    coef_s = prem_s[None, :, :] * np.exp(rf * T1) - payout_s
    coef_l = prem_l[None, :, :] * np.exp(rf * T2) - payout_l

    stock_term = S_T2 @ x
    opt_s_term = cp.sum(cp.multiply(coef_s, cp.reshape(p_s, (1, n_a, n_k), order='C')), axis=(1, 2))
    opt_l_term = cp.sum(cp.multiply(coef_l, cp.reshape(p_l, (1, n_a, n_k), order='C')), axis=(1, 2))
    r = stock_term + opt_s_term + opt_l_term - 1.0
    W = 1.0 + r

    turnover = cp.abs(w - w_prev)
    util = cp.sum(W) / n_scen - cp.sum(cp.square(W)) / (2 * s_util * n_scen)
    objective = cp.Maximize(util - turnover_cost * cp.sum(turnover))

    constraints = [cp.sum(w) == 1, w >= w_min, w <= w_max, cp.sum(turnover) <= turnover_max]
    for j in range(n_a):
        constraints.append(cp.sum(p_s[j, :]) + cp.sum(p_l[j, :]) <= x[j])

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, max_iter=20000, eps_abs=1e-5, eps_rel=1e-5, verbose=False)
    if w.value is None:
        return None
    return dict(status=prob.status, w=w.value, p_s=p_s.value, p_l=p_l.value,
                K_s=K_s, K_l=K_l, prem_s=prem_s, prem_l=prem_l)


print("موتورِ QPِ چندسررسیدی (Quadratic Utility + هزینه‌ی گردش) آماده شد.")
"""))

new_cells.append(code(r"""
N_SCEN = 800   # تعدادِ سناریوهایِ Monte Carlo برایِ simulate_two_horizon
qp_rng = np.random.default_rng(321)
qp_rows = []
qp_weight_history = []
qp_maturity_mix = []
qp_skipped = 0
w_prev_qp = np.ones(len(ASSET_NAMES)) / len(ASSET_NAMES)

for t in reb_dates_common:
    hist = ret_wide_bt.loc[:t].iloc[-BL_COV_WINDOW:]
    if len(hist) < 60:
        qp_skipped += 1; continue
    Sigma_t = _LedoitWolf2().fit(hist.values).covariance_ * 252
    corr_t = Sigma_t / np.outer(np.sqrt(np.diag(Sigma_t)), np.sqrt(np.diag(Sigma_t)))

    views_t = {name: rolling_view_at(name, t) for name in ASSET_NAMES}
    if any(v is None for v in views_t.values()):
        qp_skipped += 1; continue

    S0_vec, mu_vec, sigma_vec, sigma_bs_vec = [], [], [], []
    actual_T1, actual_T2 = [], []
    valid = True
    for name in ASSET_NAMES:
        close = processed_bt[name]['close']
        if t not in close.index:
            valid = False; break
        pos = close.index.get_loc(t)
        if pos + ROLL_DAYS >= len(close):
            valid = False; break
        S0_vec.append(close.iloc[pos])
        actual_T1.append(close.iloc[pos + SHORT_MATURITY_DAYS])
        actual_T2.append(close.iloc[pos + ROLL_DAYS])
        mu_vec.append(views_t[name]['mu_ann']); sigma_vec.append(views_t[name]['sigma_ann'])
        h60 = close.loc[:t].pct_change().dropna().iloc[-60:]
        sigma_bs_vec.append(h60.std() * np.sqrt(252) if len(h60) > 10 else views_t[name]['sigma_ann'])
    if not valid:
        qp_skipped += 1; continue

    S0_vec, mu_vec, sigma_vec, sigma_bs_vec, actual_T1, actual_T2 = map(
        np.array, (S0_vec, mu_vec, sigma_vec, sigma_bs_vec, actual_T1, actual_T2))
    T1, T2 = SHORT_MATURITY_DAYS / TRADING_DAYS_PER_YEAR, ROLL_DAYS / TRADING_DAYS_PER_YEAR

    sol = solve_multi_maturity_qp(S0_vec, mu_vec, sigma_vec, sigma_bs_vec, corr_t, T1, T2, OTM_GRID,
                                   RISK_FREE_RATE, BL_W_MIN, BL_W_MAX, S_UTIL, TURNOVER_COST, TURNOVER_MAX,
                                   w_prev_qp, N_SCEN, qp_rng)
    if sol is None:
        qp_skipped += 1; continue

    x_sol = sol['w'] / S0_vec
    realized_r = float(np.dot(x_sol, actual_T2)) - 1.0
    for j in range(len(ASSET_NAMES)):
        for l in range(len(OTM_GRID)):
            payout_s_ijl = max(actual_T1[j] - sol['K_s'][j, l], 0.0)
            realized_r += sol['p_s'][j, l] * (sol['prem_s'][j, l] * np.exp(RISK_FREE_RATE * T1) - payout_s_ijl)
            payout_l_ijl = max(actual_T2[j] - sol['K_l'][j, l], 0.0)
            realized_r += sol['p_l'][j, l] * (sol['prem_l'][j, l] * np.exp(RISK_FREE_RATE * T2) - payout_l_ijl)
    turnover_realized = float(np.abs(sol['w'] - w_prev_qp).sum())
    realized_r -= TURNOVER_COST * turnover_realized

    bh_r = float(np.dot(sol['w'], actual_T2 / S0_vec - 1.0))

    short_wealth = float((sol['p_s'].sum(axis=1) * S0_vec).sum())
    long_wealth = float((sol['p_l'].sum(axis=1) * S0_vec).sum())

    qp_rows.append(dict(date=t, cc_ret=realized_r, bh_ret=bh_r, year=t.year, turnover=turnover_realized,
                         short_maturity_share=short_wealth / max(short_wealth + long_wealth, 1e-9)))
    qp_weight_history.append(dict(date=t, **{name: sol['w'][j] for j, name in enumerate(ASSET_NAMES)}))
    w_prev_qp = sol['w']

qp_df = pd.DataFrame(qp_rows).set_index('date')
qp_weight_history_df = pd.DataFrame(qp_weight_history).set_index('date')
print(f"✅ بک‌تستِ QPِ چندسررسیدی اجرا شد — {len(qp_df)} دوره ({qp_skipped} رد شد).")
print(f"میانگینِ سهمِ ارزشِ پوشش‌داده‌شده با سررسیدِ کوتاه (۱۰روزه): {qp_df['short_maturity_share'].mean():.1%}")
print(f"میانگینِ گردشِ پرتفو در هر rebalance: {qp_df['turnover'].mean():.1%}")

qp_perf = perf_metrics2(qp_df['cc_ret'], periods_per_year_corrected)
qp_bh_perf = perf_metrics2(qp_df['bh_ret'], periods_per_year_corrected)
qp_ci = bootstrap_ci(qp_df['cc_ret'], periods_per_year_corrected)
print(f"\nMultiMaturity_QP_Monthly: TotalReturn={qp_perf['TotalReturn']:.1%} Sharpe={qp_perf['Sharpe']:.2f} "
      f"(CI90: {qp_ci['TotalReturn_lo']:.1%} تا {qp_ci['TotalReturn_hi']:.1%})")
print(f"Buy&Hold (همان وزن‌ها): TotalReturn={qp_bh_perf['TotalReturn']:.1%} Sharpe={qp_bh_perf['Sharpe']:.2f}")

portfolio_compare_rows.append(dict(Variant='MultiMaturity_QP_Monthly (چندسررسیدی+Quadratic Utility+Turnover)',
                                    TotalReturn=qp_perf['TotalReturn'], Sharpe=qp_perf['Sharpe'],
                                    TR_CI90_lo=qp_ci['TotalReturn_lo'], TR_CI90_hi=qp_ci['TotalReturn_hi']))
portfolio_compare_df = pd.DataFrame(portfolio_compare_rows)
print()
print(portfolio_compare_df.round(3).to_string(index=False))
"""))


new_cells.append(md(r"""
---
# بخشِ ۶ — Combinatorial Purged Cross-Validation (CPCV)
---

## چرا این بخش لازم شد؟

نگرانی‌ای که از Deflated Sharpe Ratio (بخشِ ۴ب-۳) بیرون آمد این بود که Sharpe=۰.۷۵
از **یک** تقسیمِ تکیِ زمانی به دست آمده — یعنی نمی‌دانیم این عدد پایدار است یا
محصولِ همان یک برشِ خاص از تاریخ. طبقِ López de Prado (پیشرفتِ ۲۰۲۴ روی روشِ
اصلیِ او)، **CPCV** به‌جایِ یک برش، دوره‌هایِ Test را به چند بلوکِ زمانی تقسیم
می‌کند و **همه‌یِ ترکیب‌هایِ ممکنِ** انتخابِ نیمی از بلوک‌ها را می‌آزماید — به
این ترتیب به‌جایِ یک عددِ Sharpe، یک **توزیع** از Sharpe به دست می‌آید.

⚠️ **صادقانه:** نسخه‌یِ کاملِ CPCV نیازمندِ آموزشِ دوباره‌یِ مدل‌ها روی هرکدام
از ده‌ها ترکیبِ ممکن است — با مقیاسِ این پروژه (بازآموزیِ ۹ مدل × ۶ سهم برایِ
هر ترکیب) کاملاً غیرِعملی است. در عوض، منطقِ ترکیبیِ CPCV را روی **سریِ
بازدهِ خارج‌از-نمونه‌یِ از‌پیش‌محاسبه‌شده‌یِ** `Rolling_BL_Monthly` (که خودش
Walk-Forward و بدونِ نشتی است) پیاده می‌کنیم — یعنی می‌پرسیم: «اگر فقط زیرمجموعه‌ای
از همین دوره‌هایِ تاریخی را می‌دیدیم، باز هم به همین جمع‌بندی می‌رسیدیم؟»
"""))

new_cells.append(code(r"""
from itertools import combinations as _combinations

CPCV_N_BLOCKS = 8
CPCV_K_INCLUDE = 4   # نیمی از بلوک‌ها در هر ترکیب انتخاب می‌شوند

pcc_roll_arr = np.asarray(pcc_roll)
block_edges = np.linspace(0, len(pcc_roll_arr), CPCV_N_BLOCKS + 1).astype(int)
blocks = [pcc_roll_arr[block_edges[i]:block_edges[i + 1]] for i in range(CPCV_N_BLOCKS)]

cpcv_rows = []
for combo in _combinations(range(CPCV_N_BLOCKS), CPCV_K_INCLUDE):
    sub_returns = np.concatenate([blocks[i] for i in combo])
    if len(sub_returns) < 3:
        continue
    perf = perf_metrics2(sub_returns, periods_per_year_corrected)
    cpcv_rows.append(dict(Combo=str(combo), N=perf['N'], TotalReturn=perf['TotalReturn'], Sharpe=perf['Sharpe']))

cpcv_df = pd.DataFrame(cpcv_rows)
frac_negative_sharpe = float((cpcv_df['Sharpe'] < 0).mean())
frac_negative_return = float((cpcv_df['TotalReturn'] < 0).mean())

print(f"تعدادِ ترکیب‌هایِ CPCV: C({CPCV_N_BLOCKS},{CPCV_K_INCLUDE}) = {len(cpcv_df)}")
print(f"\nآماره‌هایِ توزیعِ Sharpe رویِ همه‌یِ ترکیب‌ها:")
print(cpcv_df['Sharpe'].describe().round(3))
print(f"\nSharpeِ تک‌عددیِ اصلی (کلِ دوره، بدونِ CPCV): {perf_metrics2(pcc_roll_arr, periods_per_year_corrected)['Sharpe']:.3f}")
print(f"\nسهمِ ترکیب‌هایی با Sharpeِ منفی: {frac_negative_sharpe:.1%}")
print(f"سهمِ ترکیب‌هایی با بازدهِ کلِ منفی: {frac_negative_return:.1%}")
print("\nتفسیر: اگر این سهم بالا باشد (مثلاً >30-40%)، یعنی نتیجه‌ی مثبتِ اصلی به یک")
print("زیرمجموعه‌ی خاص از تاریخ حساس است، نه یک الگویِ پایدار در کلِ داده.")
"""))

new_cells.append(md(r"""
## همان تحلیل رویِ نسخه‌یِ MultiMaturity_QP_Monthly (برایِ مقایسه)
"""))

new_cells.append(code(r"""
qp_ret_arr = np.asarray(qp_df['cc_ret'])
block_edges_qp = np.linspace(0, len(qp_ret_arr), CPCV_N_BLOCKS + 1).astype(int)
blocks_qp = [qp_ret_arr[block_edges_qp[i]:block_edges_qp[i + 1]] for i in range(CPCV_N_BLOCKS)]

cpcv_qp_rows = []
for combo in _combinations(range(CPCV_N_BLOCKS), CPCV_K_INCLUDE):
    sub_returns = np.concatenate([blocks_qp[i] for i in combo])
    if len(sub_returns) < 3:
        continue
    perf = perf_metrics2(sub_returns, periods_per_year_corrected)
    cpcv_qp_rows.append(dict(Combo=str(combo), N=perf['N'], TotalReturn=perf['TotalReturn'], Sharpe=perf['Sharpe']))

cpcv_qp_df = pd.DataFrame(cpcv_qp_rows)
print("MultiMaturity_QP_Monthly — توزیعِ Sharpe رویِ ترکیب‌هایِ CPCV:")
print(cpcv_qp_df['Sharpe'].describe().round(3))
print(f"سهمِ ترکیب‌هایی با Sharpeِ منفی: {(cpcv_qp_df['Sharpe'] < 0).mean():.1%}")
"""))

new_cells.append(md(r"""
## جمع‌بندیِ CPCV
"""))

new_cells.append(code(r"""
print("="*70)
print("جمع‌بندیِ CPCV — پایداریِ نتیجه‌ی Rolling_BL_Monthly در برابرِ زیرنمونه‌گیریِ ترکیبی")
print("="*70)
print(f"Sharpeِ میانگین رویِ {len(cpcv_df)} ترکیب: {cpcv_df['Sharpe'].mean():.3f} "
      f"(انحرافِ معیار: {cpcv_df['Sharpe'].std():.3f})")
print(f"بازه‌یِ Sharpe رویِ ترکیب‌ها: {cpcv_df['Sharpe'].min():.3f} تا {cpcv_df['Sharpe'].max():.3f}")
print(f"سهمِ ترکیب‌هایی با نتیجه‌یِ منفی: {frac_negative_sharpe:.1%}")
if frac_negative_sharpe > 0.30:
    print("\n⚠️ نتیجه: بخشِ قابلِ‌توجهی از ترکیب‌هایِ ممکن Sharpeِ منفی نشان می‌دهند —")
    print("   یعنی نتیجه‌یِ مثبتِ اصلی به انتخابِ خاصِ زیرمجموعه‌یِ تاریخی حساس است.")
else:
    print("\n✅ نتیجه: اکثریتِ ترکیب‌ها همچنان Sharpeِ مثبت نشان می‌دهند —")
    print("   نشانه‌یِ نسبیِ پایداریِ الگو در زیرنمونه‌هایِ مختلف (نه اثباتِ قطعی).")
"""))

new_cells.append(md("## ذخیره‌ی خروجیِ بک‌تستِ اصلاح‌شده"))

new_cells.append(code(r"""
corrected_summary_df.to_csv(DATA_DIR + 'backtest_corrected_results.csv', index=False)
subperiod_df.to_csv(DATA_DIR + 'backtest_corrected_subperiods.csv', index=False)
portfolio_compare_df.to_csv(DATA_DIR + 'backtest_corrected_portfolio_comparison.csv', index=False)
rf_sensitivity_df.to_csv(DATA_DIR + 'backtest_rf_sensitivity.csv', index=False)
weight_history_df.to_csv(DATA_DIR + 'backtest_rolling_bl_weight_history.csv')
for name in ASSET_NAMES:
    rolling_results[name].to_csv(DATA_DIR + f'backtest_rolling_bl_{name}.csv')
dsr_df.to_csv(DATA_DIR + 'backtest_deflated_sharpe.csv', index=False)
price_limit_df.to_csv(DATA_DIR + 'backtest_price_limit_feasibility.csv', index=False)
threshold_sensitivity_df.to_csv(DATA_DIR + 'backtest_corp_action_threshold_sensitivity.csv', index=False)
qp_df.to_csv(DATA_DIR + 'backtest_multimaturity_qp_monthly.csv')
qp_weight_history_df.to_csv(DATA_DIR + 'backtest_multimaturity_qp_weight_history.csv')
cpcv_df.to_csv(DATA_DIR + 'backtest_cpcv_rolling_bl.csv', index=False)
cpcv_qp_df.to_csv(DATA_DIR + 'backtest_cpcv_multimaturity_qp.csv', index=False)
predicted_vs_actual_df.to_csv(DATA_DIR + 'backtest_predicted_vs_actual.csv', index=False)
print("ذخیره شد: backtest_corrected_results.csv, backtest_corrected_subperiods.csv, "
      "backtest_corrected_portfolio_comparison.csv, backtest_rf_sensitivity.csv, "
      "backtest_rolling_bl_weight_history.csv, backtest_rolling_bl_<Asset>.csv, "
      "backtest_deflated_sharpe.csv, backtest_price_limit_feasibility.csv, "
      "backtest_corp_action_threshold_sensitivity.csv, "
      "backtest_multimaturity_qp_monthly.csv, backtest_multimaturity_qp_weight_history.csv, "
      "backtest_cpcv_rolling_bl.csv, backtest_cpcv_multimaturity_qp.csv, "
      "backtest_predicted_vs_actual.csv")
"""))

# همیشه درست قبل از آخرین سلول (که «جمع‌بندیِ نهایی» است) درج می‌شود — نه با
# ایندکسِ عددیِ ثابت، تا رشدِ تعدادِ سلول‌هایِ بخش‌های قبلی (مثلِ Part 1) این را خراب نکند.
insert_at = len(nb.cells) - 1
nb.cells = nb.cells[:insert_at] + new_cells + nb.cells[insert_at:]

nbformat.write(nb, NB_PATH)
print(f"Inserted {len(new_cells)} cells at position {insert_at}. New total: {len(nb.cells)}")
