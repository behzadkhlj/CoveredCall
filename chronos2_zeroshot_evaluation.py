import time, warnings, torch, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from chronos import BaseChronosPipeline

DATA_DIR = 'data/'
CORP_ACTION_THRESHOLD = 0.25
ASSET_NAMES = ['Fameli', 'Fulad', 'IranKhodro', 'Khgostar', 'Shapna', 'VebMellat']

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

pipeline = BaseChronosPipeline.from_pretrained('models/chronos-bolt-small', device_map='cpu', torch_dtype=torch.float32)

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
    for _, row in pred.iterrows():
        t = row['date']
        hist = close.loc[:t]
        contexts.append(torch.tensor(hist.values, dtype=torch.float32))

    t0 = time.time()
    quantiles, mean = pipeline.predict_quantiles(inputs=contexts, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9], batch_size=64)
    elapsed = time.time() - t0

    chronos_pred = np.array([q[0, H-1, 1].item() for q in quantiles])
    chronos_q10 = np.array([q[0, H-1, 0].item() for q in quantiles])
    chronos_q90 = np.array([q[0, H-1, 2].item() for q in quantiles])

    actual = pred['actual_price'].values
    s0 = np.array([close.loc[:row['date']].iloc[-1] for _, row in pred.iterrows()])

    rmse_price = float(np.sqrt(np.mean((chronos_pred - actual) ** 2)))
    mae_price = float(np.mean(np.abs(chronos_pred - actual)))
    mape = float(np.mean(np.abs((chronos_pred - actual) / actual)) * 100)
    logret_pred = np.log(chronos_pred / s0)
    logret_actual = np.log(actual / s0)
    rmse_logret = float(np.sqrt(np.mean((logret_pred - logret_actual) ** 2)))
    dir_acc = float(np.mean(np.sign(logret_pred) == np.sign(logret_actual)))
    ss_res = np.sum((actual - chronos_pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    coverage80 = float(np.mean((actual >= chronos_q10) & (actual <= chronos_q90)))

    summary_rows.append(dict(Asset=name, Model='Chronos2_ZeroShot', Horizon=H,
                              RMSE_logret=rmse_logret, MAE_price=mae_price, RMSE_price=rmse_price,
                              MAPE_pct=mape, DirectionalAcc=dir_acc, R2=r2,
                              Quantile80_Coverage=coverage80, N_test=len(pred), Infer_sec=elapsed))
    all_preds[name] = pd.DataFrame({'date': pred['date'], 'actual_price': actual,
                                     'chronos_pred': chronos_pred, 'chronos_q10': chronos_q10, 'chronos_q90': chronos_q90})
    print(f"{name}: H={H} N={len(pred)} RMSE_price={rmse_price:.1f} MAPE={mape:.2f}% DirAcc={dir_acc:.3f} R2={r2:.3f} time={elapsed:.1f}s")

print(f"\nTotal time: {time.time()-t0_all:.1f}s")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(DATA_DIR + 'chronos2_zeroshot_results.csv', index=False)
for name, dfp in all_preds.items():
    dfp.to_csv(DATA_DIR + f'chronos2_zeroshot_predictions_{name}.csv', index=False)

print("\n=== Chronos-2 vs existing best models (Naive_RW, Ensemble) ===")
for name in ASSET_NAMES:
    naive = results_existing[(results_existing.Asset == name) & (results_existing.Model == 'Naive_RW')].iloc[0]
    ens = results_existing[(results_existing.Asset == name) & (results_existing.Model == 'Ensemble')].iloc[0]
    ch = summary_df[summary_df.Asset == name].iloc[0]
    print(f"\n{name} (H={ch['Horizon']}d):")
    print(f"  Naive_RW    RMSE_price={naive['RMSE_price']:.1f}  RMSE_logret={naive['RMSE_logret']:.4f}  MAPE={naive['MAPE_pct']:.2f}%")
    print(f"  Ensemble    RMSE_price={ens['RMSE_price']:.1f}  RMSE_logret={ens['RMSE_logret']:.4f}  MAPE={ens['MAPE_pct']:.2f}%")
    print(f"  Chronos2    RMSE_price={ch['RMSE_price']:.1f}  RMSE_logret={ch['RMSE_logret']:.4f}  MAPE={ch['MAPE_pct']:.2f}%  <-- beats Naive: {ch['RMSE_logret'] < naive['RMSE_logret']}")

print("\nSaved: data/chronos2_zeroshot_results.csv, data/chronos2_zeroshot_predictions_<Asset>.csv")
