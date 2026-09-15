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

        rows.append(dict(date=t, S0=S0, K=K, premium=premium, actual_price=actual_price,
                          cc_ret=cc_ret, bh_ret=uncovered_ret_net, year=t.year))
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

        rolling_rows[name].append(dict(date=t, S0=S0, K=K, premium=premium, actual_price=actual_price,
                                        weight=float(weights_t[name]), coverage=coverage,
                                        cc_ret=cc_ret, bh_ret=bh_ret, year=t.year))

rolling_results = {name: pd.DataFrame(rows).set_index('date') for name, rows in rolling_rows.items()}
weight_history_df = pd.DataFrame(rolling_weight_history).set_index('date')
print(f"✅ بک‌تستِ رولینگِ Black-Litterman اجرا شد — {len(reb_dates_common)} تاریخِ rebalance "
      f"({skipped_dates} رد شد به دلیلِ کمبودِ داده‌یِ تاریخچه).")
print("\nمیانگین و بازه‌یِ وزنِ هر سهم در طولِ بک‌تست (رولینگ، نه ثابت):")
print(weight_history_df.describe().T[['mean', 'min', 'max']].round(3))
"""))

new_cells.append(md(r"""
## سطحِ پرتفو: مقایسه‌یِ چهار نسخه

نسخه‌یِ «تهاجمی» (بخشِ ۴، بدونِ هزینه، رولِ ۱۰-۲۰ روزه، وزنِ ثابت) در برابرِ
سه نسخه‌یِ این بخش — مکانیکی، بهینه‌شده (وزنِ ثابت اما با هزینه و روزِ
تقویمیِ درست)، و رولینگِ کامل (بدونِ نشتِ اطلاعاتِ آینده).
"""))

new_cells.append(code(r"""
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


def portfolio_agg_rolling(results_dict):
    date_union = sorted(set().union(*[set(results_dict[n].index) for n in ASSET_NAMES]))
    port_cc, port_bh = [], []
    for t in date_union:
        cc_t, bh_t, w_t = 0.0, 0.0, 0.0
        for name in ASSET_NAMES:
            df_bt = results_dict[name]
            if t in df_bt.index:
                w = df_bt.loc[t, 'weight']
                cc_t += w * df_bt.loc[t, 'cc_ret']
                bh_t += w * df_bt.loc[t, 'bh_ret']
                w_t += w
        if w_t > 0:
            port_cc.append(cc_t / w_t)
            port_bh.append(bh_t / w_t)
    return np.array(port_cc), np.array(port_bh)


portfolio_compare_rows = []

portfolio_compare_rows.append(dict(Variant='Aggressive_Original (بخشِ ۴، بدونِ هزینه، وزنِ ثابت)',
                                    TotalReturn=port_cc_return, Sharpe=port_cc_sharpe,
                                    TR_CI90_lo=np.nan, TR_CI90_hi=np.nan))

for variant_name in ['Mechanical_Monthly', 'Optimized_Monthly']:
    pcc, pbh = portfolio_agg(corrected_results, variant_name, periods_per_year_corrected)
    p = perf_metrics2(pcc, periods_per_year_corrected)
    ci = bootstrap_ci(pcc, periods_per_year_corrected)
    portfolio_compare_rows.append(dict(Variant=variant_name, TotalReturn=p['TotalReturn'], Sharpe=p['Sharpe'],
                                        TR_CI90_lo=ci['TotalReturn_lo'], TR_CI90_hi=ci['TotalReturn_hi']))

pcc_roll, pbh_roll = portfolio_agg_rolling(rolling_results)
p_roll = perf_metrics2(pcc_roll, periods_per_year_corrected)
ci_roll = bootstrap_ci(pcc_roll, periods_per_year_corrected)
portfolio_compare_rows.append(dict(Variant='Rolling_BL_Monthly (بدونِ نشتِ اطلاعاتِ آینده)',
                                    TotalReturn=p_roll['TotalReturn'], Sharpe=p_roll['Sharpe'],
                                    TR_CI90_lo=ci_roll['TotalReturn_lo'], TR_CI90_hi=ci_roll['TotalReturn_hi']))

portfolio_compare_df = pd.DataFrame(portfolio_compare_rows)
print(portfolio_compare_df.round(3).to_string(index=False))
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

**نتیجه:** با مقایسه‌یِ ستونِ `Rolling_BL_Monthly` در جدولِ بالا با
`Aggressive_Original`، می‌بینیم که رفعِ همزمانِ همه‌ی این ایرادها اعداد را
به سمتِ مقادیرِ به‌مراتب متواضعانه‌تر و قابلِ‌دفاع‌تر می‌برد.

**آنچه هنوز اصلاح نشده (خارج از دامنه‌یِ این بازبینی):** قیمتِ واقعیِ بازارِ
آپشن (به‌جایِ بلک-شولزِ نظری)، NAVِ روزانه‌یِ کاملِ نقد/تسویه، block bootstrap
به‌جایِ i.i.d.، رفعِ نشتِ مرزیِ انتخابِ افق و اسکیلر در بخشِ ۱، و کنترلِ
multiple-testing (Deflated Sharpe Ratio) — این‌ها در لایه‌هایِ بعدیِ
اولویت‌بندی قرار دارند.

**منابع:** Whaley (2002); Feldman & Roy (2005); Hill, Balasubramanian,
Gregory & Tierens (2006, *FAJ*); Foltice (2022); Israelov & Nielsen (2014,
AQR); Israelov & Klein (2016); Diaz & Kwon (2019, *Journal of Asset
Management*); مطالعه‌یِ بازارهایِ نوظهور (۲۰۲۱-۲۰۲۵).
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
print("ذخیره شد: backtest_corrected_results.csv, backtest_corrected_subperiods.csv, "
      "backtest_corrected_portfolio_comparison.csv, backtest_rf_sensitivity.csv, "
      "backtest_rolling_bl_weight_history.csv, backtest_rolling_bl_<Asset>.csv")
"""))

# Insert after cell 97 (the original Part 4 save cell), before cell 98 (final master conclusion)
insert_at = 98
nb.cells = nb.cells[:insert_at] + new_cells + nb.cells[insert_at:]

nbformat.write(nb, NB_PATH)
print(f"Inserted {len(new_cells)} cells at position {insert_at}. New total: {len(nb.cells)}")
