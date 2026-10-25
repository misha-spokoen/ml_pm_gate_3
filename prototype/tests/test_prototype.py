"""Meaningful regressions for data lineage, holdout and monetary guardrails."""
import json
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT, FEATURES, eligibility, feature_frame, quote, select_campaign


class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshots=pd.read_csv(ROOT/"data/snapshots.csv.gz",parse_dates=["cutoff","feature_max_date","label_available"])
        cls.pred=pd.read_csv(ROOT/"results/predictions.csv.gz")
        cls.campaign=pd.read_csv(ROOT/"results/campaign.csv")
        cls.panel=np.load(ROOT/"data/monthly_panel.npz")
        with (ROOT/"models/bundle.pkl").open("rb") as f: cls.models=pickle.load(f)

    def test_features_reproduce_from_pre_cutoff_panel(self):
        """Rebuild a real feature from stored source observations, not date metadata."""
        d=self.snapshots
        cuts=(d.cutoff.dt.year-2024)*12+d.cutoff.dt.month-1
        pos=np.arange(len(d)); p=self.panel["turnover"]
        previous=np.stack([p[pos,cuts-j] for j in [2,3,4]],axis=1).mean(axis=1)
        reconstructed=p[pos,cuts-1]/np.maximum(previous,1)
        np.testing.assert_allclose(reconstructed,d.turnover_ratio,rtol=3e-7,atol=3e-7)
        self.assertTrue((d.feature_max_date<d.cutoff).all())

    def test_future_mutation_cannot_change_reconstructed_feature(self):
        d=self.snapshots.iloc[:100]; p=self.panel["turnover"][:100].copy()
        cuts=((d.cutoff.dt.year-2024)*12+d.cutoff.dt.month-1).to_numpy()
        def build(a):
            return np.array([a[i,c-1]/max(a[i,c-4:c-1].mean(),1) for i,c in enumerate(cuts)])
        before=build(p)
        for i,c in enumerate(cuts): p[i,c:]=1e12
        np.testing.assert_array_equal(before,build(p))

    def test_entities_and_label_maturity_do_not_cross_splits(self):
        d=self.snapshots
        self.assertTrue(d.client_id.is_unique)
        for a,b in [("train","calibration"),("calibration","test")]:
            self.assertLess(d.loc[d.split.eq(a),"label_available"].max(),d.loc[d.split.eq(b),"cutoff"].min())
        self.assertEqual(self.panel["turnover"].shape,(50_000,24))

    def test_forbidden_columns_do_not_enter_model(self):
        d=self.pred.iloc[:20].copy()
        changed=d.copy()
        for col in ["y_close","y_quiet","y_any","retained","arm","close_days"]: changed[col]=999
        self.assertEqual(feature_frame(d).columns.tolist(),FEATURES)
        np.testing.assert_array_equal(self.models["risk"].predict(d)["any"],self.models["risk"].predict(changed)["any"])

    def test_union_probability_contains_each_target(self):
        p=self.models["risk"].predict(self.pred.iloc[:300])
        self.assertTrue((p["any"]>=p["close"]).all())
        self.assertTrue((p["any"]>=p["quiet"]).all())
        self.assertTrue((p["any"]<=1).all())

    def test_budget_holdout_and_determinism(self):
        r=self.pred[["client_id","offer","budget_cost","saving_year","predicted_uplift","predicted_revenue"]]
        for budget in [0,300_000,3_000_000,6_000_000]:
            a=select_campaign(r,budget); b=select_campaign(r,budget)
            pd.testing.assert_frame_equal(a,b)
            self.assertTrue(a.client_id.is_unique)
            self.assertLessEqual(a.budget_cost.sum(),budget)
            self.assertEqual(int(a.group.eq("control").sum()),int(np.ceil(.1*len(a))))
            self.assertLessEqual(a.loc[a.group.eq("treatment"),"budget_cost"].sum(),budget)

    def test_disallowed_clients_never_enter_campaign(self):
        allowed=set(self.snapshots.loc[eligibility(self.snapshots),"client_id"])
        self.assertTrue(set(self.campaign.client_id)<=allowed)

    def test_price_math_and_recurring_price_trap(self):
        q=quote("advanced",50,1)
        self.assertEqual(q["before_year"],37_800)
        self.assertEqual(q["after_year"],32_790)
        self.assertEqual(q["saving_year"],5010)
        self.assertEqual(quote("advanced",50,3)["saving_year"],4060)
        self.assertFalse(quote("advanced",180,2)["allowed"])
        # An initially free package becomes more expensive after month three.
        self.assertFalse(quote("advanced",15,1)["allowed"])

    def test_drift_really_blocks_selection(self):
        with self.assertRaisesRegex(ValueError,"blocked"):
            select_campaign(pd.DataFrame(),drift_psi=.21)

    def test_excluded_client_returns_without_models(self):
        from app import Demo
        app=Demo()
        excluded=app.snapshots.loc[app.snapshots.compliance,"client_id"].iloc[0]
        class NeverScore:
            def predict(self,*args): raise AssertionError("Excluded client reached model")
        app.models={"risk":NeverScore(),"uplift":NeverScore()}
        result=app.client(excluded)
        self.assertFalse(result["eligible"])
        self.assertNotIn("risks",result)

    def test_public_duplicate_groups_are_disjoint(self):
        metrics=json.loads((ROOT/"results/public_metrics.json").read_text())
        self.assertEqual(metrics["group_overlap"],0)
        self.assertEqual(metrics["train"]+metrics["test"],3150)


if __name__=="__main__": unittest.main()
