import time, warnings, torch, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from transformers import TimesFmModelForPrediction

DATA_DIR = 'data/'
CORP_ACTION_THRESHOLD = 0.25
ASSET_NAMES = ['Fameli', 'Fulad', 'IranKhodro', 'Khgostar', 'Shapna', 'VebMellat']
BATCH_SIZE = 32

def load_and_clean(name):
    df = pd.read_csv(DATA_DIR + f'{name}.csv')
    df.columns = [c.replace('<', '').replace('>', '').strip().lower() for c in df.columns]
    df['dtyyyymmdd'] = pd.to_datetime(df['dtyyyymmdd'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['dtyyyymmdd']).set_index('dtyyyymmdd').sort_index()
    for c in ['close', 'open', 'high', 'low', 'vol', 'value', 'last']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
    return df[~df.index.duplicated(keep='last')]

def adjust_corporate_actions(df, price_cols=('close', 'open', 'high', 'low', 'last'), threshold=CORP_ACTION_THRESHOLD, ref_col='close'):
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

rec = pd.read_csv(DATA_DIR + 'price_at_maturity_recommended_models.csv').set_index('Asset')
results_existing = pd.read_csv(DATA_DIR + 'price_at_maturity_results.csv')

model = TimesFmModelForPrediction.from_pretrained('models/timesfm', torch_dtype=torch.float32)
model.eval()
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Q10_IDX = QUANTILES.index(0.1) + 1
Q50_IDX = QUANTILES.index(0.5) + 1
Q90_IDX = QUANTILES.index(0.9) + 1

summary_rows = []
all_preds = {}
t0_all = time.time()
for name in ASSET_NAMES:
    H = int(rec.loc[name, 'Horizon_days'])
    df = load_and_clean(name)
    df = adjust_corporate_actions(df)
    close = df['close']

    pred = pd.read_csv(DATA_DIR + f'price_at_maturity_predictions_{name}.csv', parse_dates=['date'])

    contexts = []
    s0_list = []
    for _, row in pred.iterrows():
        t = row['date']
        hist = close.loc[:t]
        contexts.append(torch.tensor(hist.values, dtype=torch.float32))
        s0_list.append(hist.iloc[-1])

    t0 = time.time()
    tfm_pred, tfm_q10, tfm_q90 = [], [], []
    with torch.no_grad():
        for i in range(0, len(contexts), BATCH_SIZE):
            batch_ctx = contexts[i:i + BATCH_SIZE]
            freq = torch.tensor([0] * len(batch_ctx), dtype=torch.long)
            out = model(past_values=batch_ctx, freq=freq, return_dict=True)
            fp = out.full_predictions  # (batch, horizon, 10)
            tfm_pred.extend(fp[:, H - 1, Q50_IDX].tolist())
            tfm_q10.extend(fp[:, H - 1, Q10_IDX].tolist())
            tfm_q90.extend(fp[:, H - 1, Q90_IDX].tolist())
    elapsed = time.time() - t0

    tfm_pred = np.array(tfm_pred)
    tfm_q10 = np.array(tfm_q10)
    tfm_q90 = np.array(tfm_q90)
    actual = pred['actual_price'].values
    s0 = np.array(s0_list)

    rmse_price = float(np.sqrt(np.mean((tfm_pred - actual) ** 2)))
    mae_price = float(np.mean(np.abs(tfm_pred - actual)))
    mape = float(np.mean(np.abs((tfm_pred - actual) / actual)) * 100)
    logret_pred = np.log(np.clip(tfm_pred, 1.0, None) / s0)
    logret_actual = np.log(actual / s0)
    rmse_logret = float(np.sqrt(np.mean((logret_pred - logret_actual) ** 2)))
    dir_acc = float(np.mean(np.sign(logret_pred) == np.sign(logret_actual)))
    ss_res = np.sum((actual - tfm_pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    coverage80 = float(np.mean((actual >= tfm_q10) & (actual <= tfm_q90)))

    summary_rows.append(dict(Asset=name, Model='TimesFM2_ZeroShot', Horizon=H,
                              RMSE_logret=rmse_logret, MAE_price=mae_price, RMSE_price=rmse_price,
                              MAPE_pct=mape, DirectionalAcc=dir_acc, R2=r2,
                              Quantile80_Coverage=coverage80, N_test=len(pred), Infer_sec=elapsed))
    all_preds[name] = pd.DataFrame({'date': pred['date'], 'actual_price': actual,
                                     'timesfm_pred': tfm_pred, 'timesfm_q10': tfm_q10, 'timesfm_q90': tfm_q90})
    print(f"{name}: H={H} N={len(pred)} RMSE_price={rmse_price:.1f} MAPE={mape:.2f}% DirAcc={dir_acc:.3f} R2={r2:.3f} time={elapsed:.1f}s")

print(f"\nTotal time: {time.time()-t0_all:.1f}s")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(DATA_DIR + 'timesfm2_zeroshot_results.csv', index=False)
for name, dfp in all_preds.items():
    dfp.to_csv(DATA_DIR + f'timesfm2_zeroshot_predictions_{name}.csv', index=False)

chronos_df = pd.read_csv(DATA_DIR + 'chronos2_zeroshot_results.csv')

print("\n=== TimesFM-2.0 vs Chronos-2 vs existing best models ===")
for name in ASSET_NAMES:
    naive = results_existing[(results_existing.Asset == name) & (results_existing.Model == 'Naive_RW')].iloc[0]
    ens = results_existing[(results_existing.Asset == name) & (results_existing.Model == 'Ensemble')].iloc[0]
    ch = chronos_df[chronos_df.Asset == name].iloc[0]
    tf = summary_df[summary_df.Asset == name].iloc[0]
    print(f"\n{name} (H={tf['Horizon']}d):")
    print(f"  Naive_RW    RMSE_price={naive['RMSE_price']:.1f}  RMSE_logret={naive['RMSE_logret']:.4f}  MAPE={naive['MAPE_pct']:.2f}%")
    print(f"  Ensemble    RMSE_price={ens['RMSE_price']:.1f}  RMSE_logret={ens['RMSE_logret']:.4f}  MAPE={ens['MAPE_pct']:.2f}%")
    print(f"  Chronos-2   RMSE_price={ch['RMSE_price']:.1f}  RMSE_logret={ch['RMSE_logret']:.4f}  MAPE={ch['MAPE_pct']:.2f}%")
    print(f"  TimesFM-2   RMSE_price={tf['RMSE_price']:.1f}  RMSE_logret={tf['RMSE_logret']:.4f}  MAPE={tf['MAPE_pct']:.2f}%  <-- beats Naive: {tf['RMSE_logret'] < naive['RMSE_logret']}")

print("\nSaved: data/timesfm2_zeroshot_results.csv, data/timesfm2_zeroshot_predictions_<Asset>.csv")
