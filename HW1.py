#Math 629
#Homework 1
#Mansur Alimbayev
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from math import pi
import pandas as pd

# -------------------- Config --------------------
SEED = 7
N_SAMPLES = 4000
TEST_FRAC = 0.2
BATCH = 128
EPOCHS = 120
HIDDEN = 64
ACTS = ["tanh", "relu"]
DEPTHS = ["shallow", "deep"]
OPTS = ["adam", "sgd"]  # different training methods

torch.manual_seed(SEED); np.random.seed(SEED)

# -------------------- Data --------------------
def make_data(n=N_SAMPLES):
    X = np.random.rand(n,2).astype(np.float32)
    f1 = np.sin(2*pi*X[:,0]) * np.sin(2*pi*X[:,1])                      # regression
    f2 = (X[:,0] > X[:,1]).astype(np.float32)                           # 1 if x>y else 0 (x<=y → 0)
    return X, f1[:,None].astype(np.float32), f2[:,None].astype(np.float32)

def split(X, y, test_frac=TEST_FRAC, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    k = int(len(X)*(1-test_frac))
    tr, te = idx[:k], idx[k:]
    return X[tr], X[te], y[tr], y[te]

X, y1, y2 = make_data()
# use fixed splits for fairness
X1tr,X1te,y1tr,y1te = split(X, y1, seed=1)
X2tr,X2te,y2tr,y2te = split(X, y2, seed=1)

# -------------------- Models --------------------
class MLP(nn.Module):
    def __init__(self, depth="shallow", hidden=HIDDEN, act="relu"):
        super().__init__()
        A = {"relu": nn.ReLU(), "tanh": nn.Tanh()}[act]
        if depth == "shallow":
            self.net = nn.Sequential(nn.Linear(2, hidden), A, nn.Linear(hidden, 1))
        else:
            self.net = nn.Sequential(
                nn.Linear(2, hidden), A,
                nn.Linear(hidden, hidden), A,
                nn.Linear(hidden, 1)
            )
    def forward(self, x): return self.net(x)

def make_opt(params, name):
    if name == "adam": return torch.optim.Adam(params, lr=1e-3)
    return torch.optim.SGD(params, lr=5e-2, momentum=0.9)

# -------------------- Train/Eval --------------------
def train_reg(depth, act, opt_name):
    model = MLP(depth, HIDDEN, act)
    opt = make_opt(model.parameters(), opt_name)
    crit = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(X1tr), torch.from_numpy(y1tr)),
                    batch_size=BATCH, shuffle=True)
    for _ in range(EPOCHS):
        for xb,yb in dl:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
    with torch.no_grad():
        mse = F.mse_loss(model(torch.from_numpy(X1te)), torch.from_numpy(y1te)).item()
    return model, {"MSE": mse}

def train_clf(depth, act, opt_name):
    model = MLP(depth, HIDDEN, act)
    opt = make_opt(model.parameters(), opt_name)
    crit = nn.BCEWithLogitsLoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(X2tr), torch.from_numpy(y2tr)),
                    batch_size=BATCH, shuffle=True)
    for _ in range(EPOCHS):
        for xb,yb in dl:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
    with torch.no_grad():
        logits = model(torch.from_numpy(X2te))
        probs  = torch.sigmoid(logits).numpy().ravel()  # numerically stable
        acc    = accuracy_score(y2te.ravel(), (probs>=0.5).astype(np.float32))
        auc    = roc_auc_score(y2te.ravel(), probs)
    return model, {"ACC": acc, "AUROC": auc}

# -------------------- Experiment Loop --------------------
rows = []
models_for_viz = {}  # keep best per task for visualization

for depth in DEPTHS:
    for act in ACTS:
        for opt in OPTS:
            m1, met1 = train_reg(depth, act, opt)
            rows.append(dict(task="f1", depth=depth, activation=act, optimizer=opt, **met1))
            # track best for visualization
            key = ("f1", depth)
            if key not in models_for_viz or met1["MSE"] < models_for_viz[key][1]["MSE"]:
                models_for_viz[key] = (m1, met1)

            m2, met2 = train_clf(depth, act, opt)
            rows.append(dict(task="f2", depth=depth, activation=act, optimizer=opt, **met2))
            key = ("f2", depth)
            # prefer higher ACC; break ties with AUROC
            if key not in models_for_viz or (met2["ACC"], met2["AUROC"]) > \
               (models_for_viz[key][1]["ACC"], models_for_viz[key][1]["AUROC"]):
                models_for_viz[key] = (m2, met2)

df = pd.DataFrame(rows)
df_f1 = df[df.task=='f1'][['task','depth','activation','optimizer','MSE']].sort_values('MSE')
df_f2 = df[df.task=='f2'][['task','depth','activation','optimizer','ACC','AUROC']].sort_values('ACC', ascending=False)
print(df_f1.to_string(index=False))
print(df_f2.to_string(index=False))

# -------------------- Visualizations --------------------
def heatmaps_for_task(task_key, title_suffix):
    model, metrics = models_for_viz[task_key]
    gx, gy = np.meshgrid(np.linspace(0,1,200), np.linspace(0,1,200))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    if task_key[0] == "f1":
        true = np.sin(2*np.pi*gx) * np.sin(2*np.pi*gy)
        with torch.no_grad():
            pred = model(torch.from_numpy(grid)).numpy().reshape(gx.shape)
        # true
        plt.figure(); plt.imshow(true, origin="lower", extent=[0,1,0,1], aspect="equal"); plt.colorbar()
        plt.title(f"f1 true: sin(2πx) sin(2πy)")
        plt.xlabel("x"); plt.ylabel("y"); plt.show()
        # predicted
        plt.figure(); plt.imshow(pred, origin="lower", extent=[0,1,0,1], aspect="equal"); plt.colorbar()
        plt.title(f"f1 predicted ({title_suffix}) • MSE={metrics['MSE']:.3e}")
        plt.xlabel("x"); plt.ylabel("y"); plt.show()
    else:
        true = (gx > gy).astype(float)  # diagonal labeled 0
        with torch.no_grad():
            pred = torch.sigmoid(model(torch.from_numpy(grid))).numpy().reshape(gx.shape)
        # true
        plt.figure(); plt.imshow(true, origin="lower", extent=[0,1,0,1], aspect="equal"); plt.colorbar()
        plt.title(f"f2 true: 1[x>y] (x=y labeled 0)")
        plt.xlabel("x"); plt.ylabel("y"); plt.show()
        # predicted
        plt.figure(); plt.imshow(pred, origin="lower", extent=[0,1,0,1], aspect="equal"); plt.colorbar()
        acc = metrics['ACC']; auc = metrics['AUROC']
        plt.title(f"f2 predicted prob ({title_suffix}) • ACC={acc:.3f}, AUROC={auc:.3f}")
        plt.xlabel("x"); plt.ylabel("y"); plt.show()

# Best shallow models
heatmaps_for_task(("f1","shallow"), "best shallow")
heatmaps_for_task(("f2","shallow"), "best shallow")

# Best deep models
heatmaps_for_task(("f1","deep"), "best deep")
heatmaps_for_task(("f2","deep"), "best deep")
