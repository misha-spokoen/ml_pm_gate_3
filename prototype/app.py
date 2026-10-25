"""Local, offline demo. Run: python prototype/app.py --port 8765."""
from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from core import ROOT, FEATURES, OFFERS, eligibility, quote, recommend, select_campaign


def scalar(x):
    if isinstance(x,(np.integer,np.floating)): return x.item()
    if isinstance(x,np.bool_): return bool(x)
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,pd.Timestamp): return str(x)
    raise TypeError(type(x).__name__)


class Demo:
    def __init__(self):
        with (ROOT/"models/bundle.pkl").open("rb") as f: self.models=pickle.load(f)
        self.snapshots=pd.read_csv(ROOT/"data/snapshots.csv.gz")
        self.pred=pd.read_csv(ROOT/"results/predictions.csv.gz")
        self.recs=self.pred[["client_id","offer","budget_cost","saving_year","predicted_uplift","predicted_revenue"]]
        self.results={p.stem:json.loads(p.read_text()) for p in (ROOT/"results").glob("*.json")}
        self.public=pd.read_csv(ROOT/"results/public_predictions.csv")

    def client(self, client_id, overrides=None):
        part=self.snapshots[self.snapshots.client_id.eq(client_id)].iloc[[0]].copy()
        for key,value in (overrides or {}).items():
            if key in ["compliance","liquidation","tochka","consent"]:
                if not isinstance(value,bool): raise ValueError("A flag must be boolean")
                part[key]=value
            elif key in FEATURES or key=="days_last_offer":
                if key in ["industry","region","tariff"]:
                    if value not in {"industry":["trade","services","construction"],"region":["A","B","C"],"tariff":["advanced","professional"]}[key]:
                        raise ValueError("Unknown category")
                    part[key]=value
                else:
                    number=float(value)
                    if not np.isfinite(number) or number<0 or number>1e8: raise ValueError("Invalid number")
                    part[key]=number
        row=part.iloc[0]; allowed=bool(eligibility(part).iloc[0])
        out={"client_id":client_id,"eligible":allowed,"features":json.loads(part.to_json(orient="records"))[0]}
        out["quotes"]=[{"name":OFFERS[k],**quote(row.tariff,row.payments,k)} for k in [1,2,3]]
        if not allowed:
            out["status"]="Исключён до скоринга: ограничение, миграция, согласие или частота контактов."
            return out
        start=time.perf_counter()
        out["risks"]={k:float(v[0]) for k,v in self.models["risk"].predict(part).items()}
        uplift=self.models["uplift"].predict(part)
        out["uplift"]={str(k):float(uplift[0,k-1]) for k in [1,2,3]}
        out["recommendation"]=recommend(part,uplift).to_dict(orient="records")[0]
        shap=self.models["risk"].explain(part)[0,:-1]
        keys=np.argsort(-np.abs(shap))[:3]
        out["shap"]=[{"feature":FEATURES[k],"value":float(shap[k])} for k in keys]
        out["prediction_ms"]=(time.perf_counter()-start)*1000
        # Historical outcome is available for demonstration, never in feature_frame.
        out["outcome"]={"close":int(row.y_close),"quiet":int(row.y_quiet),"any":int(row.y_any)}
        return out

    def campaign(self,budget,drift=0.):
        if not 0<=budget<=20_000_000: raise ValueError("Budget outside demo range")
        chosen=select_campaign(self.recs,budget,drift_psi=drift)
        treatment=chosen[chosen.group.eq("treatment")]
        spend=float(treatment.budget_cost.sum())
        predicted=float(treatment.predicted_revenue.sum())
        # Plan only. These are model estimates, not realized incremental outcomes.
        return {"budget":budget,"selected":len(chosen),"treatment":len(treatment),
                "control":int(chosen.group.eq("control").sum()),"spend":spend,
                "predicted_saved_revenue":predicted,"predicted_net":predicted*.5-spend,
                "rows":json.loads(chosen.head(50).to_json(orient="records")),
                "control_ids":chosen.loc[chosen.group.eq("control"),"client_id"].head(10).tolist()}


class Handler(BaseHTTPRequestHandler):
    def send(self,obj,status=200,content_type="application/json; charset=utf-8"):
        body=obj if isinstance(obj,bytes) else json.dumps(obj,ensure_ascii=False,default=scalar,allow_nan=False).encode()
        self.send_response(status); self.send_header("Content-Type",content_type)
        self.send_header("Content-Length",str(len(body))); self.send_header("Cache-Control","no-store")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        route=urlparse(self.path); q=parse_qs(route.query)
        try:
            if route.path=="/":
                self.send((ROOT/"demo.html").read_bytes(),content_type="text/html; charset=utf-8")
            elif route.path=="/api/summary":
                self.send(self.server.demo.results)
            elif route.path=="/api/clients":
                d=self.server.demo.pred.sort_values("risk_any",ascending=False).head(40)
                self.send(json.loads(d[["client_id","risk_any","risk_close","risk_quiet","offer","predicted_uplift"]].to_json(orient="records")))
            elif route.path=="/api/client":
                self.send(self.server.demo.client(q["id"][0]))
            elif route.path=="/api/special":
                df=self.server.demo.snapshots
                self.send({"excluded":df.loc[df.compliance,"client_id"].iloc[0],
                           "false_positive":self.server.demo.results["demo_error"]["client_id"],
                           "public_examples":json.loads(self.server.demo.public.sort_values("probability",ascending=False).head(5).to_json(orient="records"))})
            elif route.path=="/api/campaign":
                self.send(self.server.demo.campaign(float(q.get("budget",[3_000_000])[0]),float(q.get("drift",[0])[0])))
            elif route.path=="/download/campaign":
                chosen=select_campaign(self.server.demo.recs,float(q.get("budget",[3_000_000])[0]))
                self.send(chosen.to_csv(index=False).encode("utf-8-sig"),content_type="text/csv; charset=utf-8")
            elif route.path=="/document":
                self.send((ROOT.parent/"gate3-prototype.md").read_bytes(),content_type="text/plain; charset=utf-8")
            else: self.send({"error":"Not found"},404)
        except (ValueError,KeyError,IndexError) as e: self.send({"error":str(e)},400)

    def do_POST(self):
        if self.path!="/api/predict": return self.send({"error":"Not found"},404)
        try:
            size=int(self.headers.get("Content-Length",0))
            if size>20_000: raise ValueError("Request too large")
            payload=json.loads(self.rfile.read(size))
            self.send(self.server.demo.client(payload["client_id"],payload.get("overrides")))
        except (ValueError,KeyError,IndexError) as e: self.send({"error":str(e)},400)

    def log_message(self,fmt,*args): pass


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--port",type=int,default=8765)
    args=parser.parse_args()
    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler); server.demo=Demo()
    print(f"Gate 3 demo: http://127.0.0.1:{args.port}",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__=="__main__": main()
