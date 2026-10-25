"""Gate 3: point-in-time churn models, simulated interventions and safe actions."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
SEED = 20261025
FEATURES = [
    "turnover_ratio", "season_adjusted_ratio", "yoy_ratio", "days_no_incoming",
    "fee_shock", "fee_burden", "self_transfer_share", "complaints", "services",
    "tenure", "industry", "region", "tariff", "payments", "cash_buffer",
]
CATEGORICAL = ["industry", "region", "tariff"]
TARIFFS = {
    "simple": {"base": 490, "included": 5, "payment": 49},
    "advanced": {"base": 1990, "included": 10, "payment": 29},
    "professional": {"base": 4990, "included": 100, "payment": 19},
}
OFFERS = {1: "Пакет комиссий", 2: "Тариф ниже", 3: "Годовая оплата"}


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x)))


def feature_frame(df):
    """Explicit allow-list: IDs, labels, future events and compliance are excluded."""
    x = df[FEATURES].copy()
    for col in CATEGORICAL:
        x[col] = x[col].astype(str)
    return x


def generate(n=50_000, seed=SEED, destination=None):
    """24 observed months per entity; each entity has exactly one prediction date.

    The scenario mechanism is specified before fitting. True effects are exported
    separately and are never supplied to a model or production policy.
    """
    rng = np.random.default_rng(seed)
    ntrain, ncal = int(n * .6), int(n * .2)
    split = np.array(["train"] * ntrain + ["calibration"] * ncal + ["test"] * (n-ntrain-ncal))
    cut = np.r_[rng.choice([6, 7, 8], ntrain), rng.choice([13, 14], ncal), np.full(n-ntrain-ncal, 20)]
    idx = np.arange(n)
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    industry = rng.integers(0, 3, n)
    region = rng.integers(0, 3, n)
    tariff = rng.choice(["advanced", "professional"], n, p=[.8, .2])
    payments = rng.integers(15, 180, n)
    shock = rng.binomial(1, .25, n)
    share = rng.beta(1.3, 3, n)
    complaints = rng.poisson(.4 + .8*shock, n)
    services = rng.integers(0, 5, n)
    cash = rng.beta(2, 2, n)
    base = rng.lognormal(14, .65, n)
    amp = np.choose(industry, [.08, .22, .35])
    season = 1 + amp[:, None]*np.cos(2*np.pi*(np.arange(24)[None, :] - 6)/12)
    panel = base[:, None]*season*rng.lognormal(0, .10, (n, 24))
    declining = rng.binomial(1, .22, n)
    for lag, factor in [(2, .82), (1, .52)]:
        panel[idx, cut-lag] *= np.where(declining, factor, 1)
    # Occasional no incoming payments: a real baseline trigger, not the target.
    zero = rng.random(n) < .018
    panel[idx[zero], cut[zero]-1] = 0
    pre = np.stack([panel[idx, cut-j] for j in [1, 2, 3]], axis=1)
    older = np.stack([panel[idx, cut-j] for j in [2, 3, 4]], axis=1).mean(axis=1)
    ratio = pre[:, 0] / np.maximum(older, 1)
    adj = ratio / (season[idx, cut-1] / np.stack([season[idx, cut-j] for j in [2,3,4]], axis=1).mean(axis=1))
    # YoY is unavailable before month 13. Missingness is explicit, not backfilled.
    yoy = np.full(n, np.nan)
    have_yoy = cut >= 13
    yoy[have_yoy] = panel[idx[have_yoy], cut[have_yoy]-1] / panel[idx[have_yoy], cut[have_yoy]-13]
    fee_burden = .15 + .6*shock + rng.beta(1.5, 3, n)
    low = np.maximum(0, 1-adj)
    pclose = sigmoid(-4.0 + 1.9*shock + 2.1*share + 2.0*low + .45*complaints - .35*services)
    pquiet = sigmoid(-3.8 + 1.4*shock + 2.6*share + 2.8*low + .45*complaints - .35*services)
    # Independent future events, followed by labels derived from the future panel.
    close = rng.random(n) < pclose
    quiet = rng.random(n) < pquiet
    close_days = rng.integers(1, 61, n)
    onset = rng.integers(0, 2, n)
    for j in range(n):
        c = cut[j]
        if quiet[j]:
            k = c + onset[j]
            ref = panel[j, k-3:k].mean()
            panel[j, k:k+2] = ref*rng.uniform(.08, .24, 2)
        if close[j]:
            k = c + int(close_days[j] > 30)
            panel[j, k:] = 0
    yquiet = np.zeros(n, dtype=int)
    for k in [0, 1]:
        event = cut+k
        ref = np.stack([panel[idx, event-j] for j in [1,2,3]], axis=1).mean(axis=1)
        # An already completed two-month quiet event is not a new target event.
        new_event = ~((panel[idx, event-1] < .3*ref) & (panel[idx, event-2] < .3*ref))
        yquiet |= ((panel[idx, event] < .3*ref) & (panel[idx, event+1] < .3*ref) & new_event).astype(int)
    cutoff = months[cut]
    compliance = rng.random(n) < .025
    liquidation = rng.random(n) < .012
    tochka = rng.random(n) < .018
    p0_churn = 1-(1-pclose)*(1-pquiet)
    pricefit = sigmoid(3*(fee_burden-.55))
    lost_cause = sigmoid(9*(share-.55))
    # Potential effects are on retention probability, bounded by available risk.
    tau1 = p0_churn*(.05 + .80*pricefit)*(1-.9*lost_cause) - .015*(1-pricefit)
    tau2 = p0_churn*(.06 + .64*pricefit)*(1-.8*lost_cause)
    tau3 = p0_churn*(.04 + .55*cash)*(1-.8*lost_cause) - .04*(1-cash)
    p0_retention = 1-p0_churn
    effects = np.stack([np.zeros(n), tau1, tau2, tau3], axis=1)
    effects = np.clip(p0_retention[:,None]+effects, 0, 1)-p0_retention[:,None]
    arm = rng.integers(0, 4, n)
    retained = (rng.random(n) < p0_retention + effects[idx, arm]).astype(int)
    df = pd.DataFrame({
        "client_id": [f"SYN-{i:06d}" for i in idx], "split": split,
        "cutoff": cutoff, "feature_max_date": cutoff-pd.Timedelta(days=1),
        "label_available": cutoff+pd.Timedelta(days=120),
        "turnover_ratio": ratio, "season_adjusted_ratio": adj, "yoy_ratio": yoy,
        "days_no_incoming": np.where(zero, rng.integers(30, 60, n), rng.integers(0, 12, n)),
        "fee_shock": shock, "fee_burden": fee_burden, "self_transfer_share": share,
        "complaints": complaints, "services": services, "tenure": rng.integers(6, 60, n),
        "industry": np.array(["trade", "services", "construction"])[industry],
        "region": np.array(["A", "B", "C"])[region], "tariff": tariff,
        "payments": payments, "cash_buffer": cash,
        "compliance": compliance, "liquidation": liquidation, "tochka": tochka,
        "consent": rng.random(n) > .08, "days_last_offer": rng.integers(0, 180, n),
        "revenue_year": np.full(n, 60_000.),
        "y_close": close.astype(int), "y_quiet": yquiet,
        "close_days": np.where(close, close_days, -1),
        "arm": arm, "retained": retained,
    })
    df["y_any"] = np.maximum(df.y_close, df.y_quiet)
    truth = pd.DataFrame({"client_id": df.client_id, "p0_retention": p0_retention,
                          **{f"tau_{k}": effects[:, k] for k in [1,2,3]}})
    if destination:
        destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination/"monthly_panel.npz", turnover=panel.astype("float32"),
                            client_id=df.client_id.to_numpy(dtype="U10"), month=months.astype(str).to_numpy(dtype="U10"))
        df.to_csv(destination/"snapshots.csv.gz", index=False)
        truth.to_csv(destination/"simulation_truth.csv.gz", index=False)
    return df, truth


def eligibility(df):
    return (~df.compliance.astype(bool) & ~df.liquidation.astype(bool) & ~df.tochka.astype(bool)
            & df.consent.astype(bool) & (df.days_last_offer >= 30) & (df.tenure >= 6)
            & df.tariff.isin(["advanced", "professional"]))


def rule_score(df):
    # Exact rule triggers. Equal priority uses the fixed seeded input order, not labels.
    return (df.days_no_incoming.ge(30).astype(int)*2 + df.turnover_ratio.lt(.5).astype(int)).to_numpy(float)


class ChurnModels:
    def fit(self, train, cal):
        self.models, self.calibrators = {}, {}
        x, xc = feature_frame(train), feature_frame(cal)
        for target in ["close", "quiet"]:
            model = CatBoostClassifier(iterations=300, depth=5, learning_rate=.06,
                                       loss_function="Logloss", random_seed=SEED,
                                       thread_count=4, verbose=False, allow_writing_files=False)
            model.fit(x, train[f"y_{target}"], cat_features=CATEGORICAL)
            raw = model.predict_proba(xc)[:,1]
            self.models[target] = model
            self.calibrators[target] = IsotonicRegression(out_of_bounds="clip").fit(raw, cal[f"y_{target}"])
        raw_union = self._raw_union(cal)
        self.union = IsotonicRegression(out_of_bounds="clip").fit(raw_union, cal.y_any)
        self.rule_probability = {}
        for target in ["close", "quiet", "any"]:
            # Rule probabilities calibrated on a disjoint calibration period too.
            self.rule_probability[target] = {int(k): float(cal.loc[rule_score(cal)==k, f"y_{target}"].mean())
                                             for k in np.unique(rule_score(cal))}
        return self

    def _raw_union(self, df):
        x = feature_frame(df)
        return sum(m.predict_proba(x)[:,1] for m in self.models.values())

    def predict(self, df):
        x = feature_frame(df)
        out = {t: self.calibrators[t].predict(m.predict_proba(x)[:,1]) for t,m in self.models.items()}
        # A union probability cannot be smaller than either marginal probability.
        # Independent calibrators do not guarantee that constraint by themselves.
        combined = self.union.predict(self._raw_union(df))
        out["any"] = np.maximum(combined, np.maximum(out["close"], out["quiet"]))
        return out

    def explain(self, df):
        model = self.models["quiet"]
        return model.get_feature_importance(Pool(feature_frame(df), cat_features=CATEGORICAL),
                                            type="ShapValues", thread_count=4)


class UpliftModels:
    """Multi-arm T-learner. Only observed randomized outcomes are used to fit."""
    def fit(self, train):
        self.models = {}
        for arm in [0,1,2,3]:
            part = train[train.arm==arm]
            model = CatBoostClassifier(iterations=320, depth=4, learning_rate=.05,
                                       l2_leaf_reg=8, loss_function="Logloss", random_seed=SEED+arm,
                                       thread_count=4, verbose=False, allow_writing_files=False)
            model.fit(feature_frame(part), part.retained, cat_features=CATEGORICAL)
            self.models[arm] = model
        return self

    def predict(self, df):
        x = feature_frame(df)
        p0 = self.models[0].predict_proba(x)[:,1]
        return np.stack([self.models[k].predict_proba(x)[:,1]-p0 for k in [1,2,3]], axis=1)


def monthly_cost(tariff, payments):
    t = TARIFFS[tariff]
    return int(t["base"] + max(0, int(payments)-t["included"])*t["payment"])


def quote(tariff, payments, offer):
    """Integer kopeck-free scenario calculator, no ML, no current-market claim.

    Guardrail checks both the first year and recurring month after the promotion.
    """
    current = monthly_cost(tariff, payments)
    t = TARIFFS[tariff]
    if offer == 1:
        saved_fees = min(max(int(payments)-t["included"], 0), 40)*t["payment"]
        recurring = current-saved_fees+990
        after_year = 12*(current-saved_fees)+9*990
        cost = 2970
    elif offer == 2:
        lower = "advanced" if tariff=="professional" else "simple"
        recurring = monthly_cost(lower, payments)
        after_year = 12*recurring
        cost = max(0, 12*(current-recurring))
    elif offer == 3:
        discount = round(t["base"]*12*.17)
        after_year = 12*current-discount
        recurring = after_year/12
        cost = discount
    else:
        raise ValueError("Unknown offer")
    return {"current_month": current, "after_month": recurring, "before_year": 12*current,
            "after_year": int(after_year), "saving_year": int(12*current-after_year),
            "budget_cost": int(cost), "allowed": bool(after_year <= 12*current and recurring <= current),
            "offer": int(offer)}


def recommend(df, uplift, margin=.5):
    out = []
    for pos, (_, row) in enumerate(df.iterrows()):
        options = []
        if bool(eligibility(df.iloc[[pos]]).iloc[0]):
            for arm in [1,2,3]:
                q = quote(row.tariff, row.payments, arm)
                effect = float(uplift[pos, arm-1])
                benefit = effect*row.revenue_year
                if q["allowed"] and effect > 0 and benefit*margin > q["budget_cost"]:
                    options.append({**q, "predicted_uplift": effect, "predicted_revenue": benefit})
        if options:
            best = max(options, key=lambda x: x["predicted_revenue"]*.5-x["budget_cost"])
        else:
            best = {"offer": 0, "budget_cost": 0, "saving_year": 0, "predicted_uplift": 0., "predicted_revenue": 0.}
        out.append({"client_id": row.client_id, **best})
    return pd.DataFrame(out)


def select_campaign(recommendations, budget=3_000_000, seed=SEED, drift_psi=0.):
    """Freeze selection first, then randomize a 10% holdout within selected clients."""
    if drift_psi > .2:
        raise ValueError("Campaign blocked: distribution shift PSI > 0.2")
    candidates = recommendations[recommendations.offer>0].sort_values(
        ["predicted_revenue", "client_id"], ascending=[False, True])
    selected, spent = [], 0
    # Reserve full cost before randomization: realized spending can only be lower.
    for ix, row in candidates.iterrows():
        if spent + row.budget_cost <= budget:
            selected.append(ix); spent += row.budget_cost
    chosen = recommendations.loc[selected].copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(chosen))
    is_control = np.zeros(len(chosen), dtype=bool)
    is_control[order[:int(np.ceil(.1*len(chosen)))]] = True
    chosen["group"] = np.where(is_control, "control", "treatment")
    return chosen


def psi(reference, current):
    reference, current = np.asarray(reference), np.asarray(current)
    bins = np.unique(np.quantile(reference, np.linspace(0,1,11)))
    bins[0], bins[-1] = -np.inf, np.inf
    a = np.histogram(reference, bins)[0].astype(float)
    b = np.histogram(current, bins)[0].astype(float)
    a = np.maximum(a/a.sum(), 1e-5); b = np.maximum(b/b.sum(), 1e-5)
    return float(np.sum((b-a)*np.log(b/a)))


def timed(call, repeats=7):
    call()  # Warm-up is reported separately from steady-state latency.
    durations = []
    for _ in range(repeats):
        start=time.perf_counter(); call(); durations.append(time.perf_counter()-start)
    return {"p50_seconds": float(np.median(durations)), "p95_seconds": float(np.quantile(durations,.95)),
            "repeats": repeats}
