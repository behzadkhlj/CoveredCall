import time, random, warnings, torch, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from chronos import BaseChronosPipeline
from peft import LoraConfig, get_peft_model

DATA_DIR = 'data/'
CORP_ACTION_THRESHOLD = 0.25
ASSET_NAMES = ['Fameli', 'Fulad', 'IranKhodro', 'Khgostar', 'Shapna', 'VebMellat']
TRAIN_CUTOFF = pd.Timestamp('2023-03-01')  # safely before earliest test start (2023-04-15), embargoes leakage
CONTEXT_LEN = 512
STRIDE = 3
BATCH_SIZE = 16
N_STEPS = 1200
LR = 1e-4
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

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

print("Building training windows (train-only, before", TRAIN_CUTOFF.date(), ")...")
pools = {}  # H -> list of (context_np, future_np)
for name in ASSET_NAMES:
    H = int(rec.loc[name, 'Horizon_days'])
    df = load_and_clean(name)
    df = adjust_corporate_actions(df)
    close = df['close']
    train_close = close[close.index < TRAIN_CUTOFF].values

    windows = []
    for i in range(0, len(train_close) - CONTEXT_LEN - H, STRIDE):
        ctx = train_close[i:i + CONTEXT_LEN]
        fut = train_close[i + CONTEXT_LEN:i + CONTEXT_LEN + H]
        windows.append((ctx.astype(np.float32), fut.astype(np.float32)))
    pools.setdefault(H, []).extend(windows)
    print(f"  {name}: H={H}, train_n={len(train_close)}, windows={len(windows)}")

for H, w in pools.items():
    print(f"Pool H={H}: {len(w)} windows total")

pipeline = BaseChronosPipeline.from_pretrained('models/chronos-bolt-small', device_map='cpu', torch_dtype=torch.float32)
model = pipeline.model
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q', 'k', 'v', 'o'], bias='none')
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
peft_model.train()
torch.set_num_threads(4)

trainable = [p for p in peft_model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=LR)

H_list = list(pools.keys())
H_weights = [len(pools[h]) for h in H_list]

loss_history = []
t0_all = time.time()
for step in range(1, N_STEPS + 1):
    H = random.choices(H_list, weights=H_weights, k=1)[0]
    num_output_patches = (H + 15) // 16
    batch = random.sample(pools[H], min(BATCH_SIZE, len(pools[H])))
    context = torch.tensor(np.stack([b[0] for b in batch]))
    future_target = torch.tensor(np.stack([b[1] for b in batch]))
    future_target_mask = torch.ones_like(future_target, dtype=torch.bool)

    out = peft_model(context=context, future_target=future_target, future_target_mask=future_target_mask, num_output_patches=num_output_patches)
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    opt.zero_grad()
    loss_history.append(loss.item())

    if step % 50 == 0 or step == 1:
        avg_recent = np.mean(loss_history[-50:])
        elapsed = time.time() - t0_all
        eta = elapsed / step * (N_STEPS - step)
        print(f"step {step}/{N_STEPS}  H={H}  loss={loss.item():.4f}  avg50={avg_recent:.4f}  elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min", flush=True)

print(f"\nTraining done. Total time: {(time.time()-t0_all)/60:.1f} min")

peft_model.eval()
merged_model = peft_model.merge_and_unload()
pipeline.model = merged_model

import os
os.makedirs('models/chronos-bolt-small-lora', exist_ok=True)
torch.save(merged_model.state_dict(), 'models/chronos-bolt-small-lora/merged_state_dict.pt')
pd.DataFrame({'step': range(1, N_STEPS + 1), 'loss': loss_history}).to_csv(DATA_DIR + 'chronos2_lora_training_loss.csv', index=False)
print("Saved merged model state_dict and training loss log.")
