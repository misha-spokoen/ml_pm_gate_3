"""Reproduce the Gate 3 experiment. Run: python prototype/run.py."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import norm, spearmanr
from sklearn.metrics import brier_score_loss, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklift.metrics import qini_auc_score, uplift_auc_score

from core import (ROOT, SEED, FEATURES, CATEGORICAL, ChurnModels, UpliftModels,
                  eligibility, feature_frame, generate, monthly_cost, psi, quote,
                  recommend, rule_score, select_campaign, timed)

OUT = ROOT/"results"


def dump(name, obj):
    def scalar(x):
        if isinstance(x, (np.integer, np.floating)): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (pd.Timestamp, Path)): return str(x)
        if isinstance(x, np.bool_): return bool(x)
        raise TypeError(type(x).__name__)
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=scalar, allow_nan=False))


def top(score, frac=.1):
    return np.argsort(-np.asarray(score), kind="stable")[:max(1, int(np.ceil(len(score)*frac)))]


def summary(y, score, probability):
    y = np.asarray(y); selected = top(score)
    p = float(y[selected].mean())
    return {"n":len(y), "positives":int(y.sum()), "k":len(selected), "prevalence":float(y.mean()),
            "precision_at_10":p, "lift_at_10":p/y.mean(),
            "recall_at_10":float(y[selected].sum()/y.sum()),
            "brier":float(brier_score_loss(y, probability)), "auc":float(roc_auc_score(y, probability))}


def boot_precision(y, a, b, seed=SEED, groups=None):
    rng=np.random.default_rng(seed); values=[]
    clusters=None if groups is None else [np.flatnonzero(np.asarray(groups)==g) for g in np.unique(groups)]
    for _ in range(400):
        ix=(rng.integers(0,len(y),len(y)) if clusters is None
            else np.concatenate([clusters[j] for j in rng.integers(0,len(clusters),len(clusters))]))
        ya=y[ix]
        pa=ya[top(a[ix])].mean(); pb=ya[top(b[ix])].mean()
        values.append(pa-pb)
    return [float(v) for v in np.quantile(values,[.025,.975])]


def calibration_bins(y, probability):
    edges=np.linspace(0,1,11); result=[]
    for low, high in zip(edges[:-1],edges[1:]):
        mask=(probability>=low)&(probability < high if high<1 else probability<=high)
        if mask.any():
            result.append({"low":low,"high":high,"n":int(mask.sum()),
                           "predicted":float(np.mean(probability[mask])),"observed":float(np.mean(np.asarray(y)[mask]))})
    return result


def evaluate_risk(test, models):
    predictions=models.predict(test)
    # Same ID tie-breaking across the baseline, ML and bootstrap.
    rs=rule_score(test)
    result={}
    for target in ["close","quiet","any"]:
        y=test[f"y_{target}"].to_numpy()
        rp=np.array([models.rule_probability[target].get(int(v), .1) for v in rs])
        a=summary(y,predictions[target],predictions[target]); b=summary(y,rs,rp)
        result[target]={"ml":a,"rules":b,
                        "precision_relative_gain":a["precision_at_10"]/b["precision_at_10"]-1,
                        "brier_relative_gain":1-a["brier"]/b["brier"],
                        "precision_difference_ci95":boot_precision(y,predictions[target],rs),
                        "calibration":calibration_bins(y,predictions[target])}
    selected=top(predictions["any"])
    days=test.iloc[selected].loc[lambda d:d.y_close.eq(1),"close_days"].to_numpy()
    result["lead_days"]={"median":float(np.median(days)),"n":len(days),
                         "note":"Only true closures in the selected top 10%; event times are scenario-generated."}
    return result,predictions


def evaluate_uplift(test, truth, model):
    predicted=model.predict(test)
    result={}
    for arm in [1,2,3]:
        part=(test.arm.isin([0,arm])).to_numpy()
        y=test.retained.to_numpy()[part]; treatment=(test.arm.to_numpy()[part]==arm).astype(int)
        score=predicted[part,arm-1]
        result[str(arm)]={"n":int(part.sum()),"treated":int(treatment.sum()),
                          "qini_normalized":float(qini_auc_score(y,score,treatment)),
                          "auuc_normalized":float(uplift_auc_score(y,score,treatment)),
                          "spearman_true_effect":float(spearmanr(predicted[:,arm-1],truth[f"tau_{arm}"]).statistic),
                          "observed_ate":float(y[treatment==1].mean()-y[treatment==0].mean()),
                          "true_ate":float(truth[f"tau_{arm}"].mean())}
    return result,predicted


def policy_comparison(test, truth, risk, uplift):
    # Fair policy comparison: identical safe package, same eligible clients, same spend.
    mask=np.array([quote(r.tariff,r.payments,1)["allowed"] for r in test.itertuples()])
    ix=np.flatnonzero(mask); k=min(1000,len(ix))
    scores={"rules":rule_score(test),"risk":risk["any"],"uplift_value":uplift[:,0]*test.revenue_year.to_numpy()}
    results={}
    for name,score in scores.items():
        chosen=ix[np.argsort(-score[ix],kind="stable")[:k]]
        ncontrol=int(np.ceil(.1*k)); ntreatment=k-ncontrol
        # Integrate over random holdout assignment to remove Monte Carlo noise
        # from comparison of similar selection policies. Operational CSV still
        # contains actual deterministic randomized group assignments.
        tau=truth.iloc[chosen].tau_1.to_numpy()
        revenue=test.iloc[chosen].revenue_year.to_numpy()
        spend=ntreatment*2970
        saved=float(np.sum(tau*revenue)*ntreatment/k)
        results[name]={"selected":k,"treatment":ntreatment,"control":ncontrol,
                       "offer_cost":2970,"spend":spend,"mean_true_uplift":float(tau.mean()),
                       "saved_revenue":saved,"revenue_per_budget":saved/spend,
                       "contribution":saved*.5,"net_before_operations":saved*.5-spend,
                       "holdout_evaluation":"Expected over random allocation of 10% control"}
    return results


def economics():
    scenarios=[]
    for uplift in [.05,.10,.15,.20]:
        for rev in [40_000,60_000,80_000]:
            n=900; cost=3000; operations=150_000; fixed=3_150_000
            saved=n*uplift*rev; contribution=saved*.5; spend=n*cost
            scenarios.append({"uplift":uplift,"revenue_year":rev,"treated":n,"spend":spend,
                              "saved_revenue":saved,"revenue_per_budget":saved/spend,
                              "net_run":contribution-spend-operations,
                              "net_first_campaign":contribution-spend-operations-fixed,
                              "roi_first_campaign":(contribution-spend-operations-fixed)/(spend+operations+fixed)})
    # Approximation with independent Bernoulli outcomes, 90/10 allocation, two-sided alpha=.05.
    p0,p1,r=.75,.85,.9
    n_total=int(np.ceil((norm.ppf(.975)+norm.ppf(.8))**2 * (p1*(1-p1)/r+p0*(1-p0)/(1-r))/(p1-p0)**2))
    return {"scenario_inputs":{"margin":.5,"offer_cost":3000,"cohort":10_000,"selected":1000,
                                "treatment":900,"control":100,"loaded_fte_month":350_000,
                                "development_fte_months":9,"development":3_150_000,
                                "monthly_operations":150_000,"year_revenue":60_000},
            "sensitivity":scenarios,"break_even_uplift":3000/(60_000*.5),
            "break_even_with_operations":(3000+150_000/900)/(60_000*.5),
            "uplift_for_product_target_3":3*3000/60_000,
            "power":{"p0_retained":p0,"p1_retained":p1,"alpha":.05,"power":.8,"allocation_treated":r,
                     "approx_n_total":n_total,"approx_n_control":int(np.ceil(n_total*(1-r))),
                     "mde_at_1000":float((norm.ppf(.975)+norm.ppf(.8))*np.sqrt(p0*(1-p0)*(1/(1000*r)+1/(1000*(1-r)))))},
            "sam_scenario":{"segment_clients":700_000,"revenue_year":60_000,"segment_revenue":42_000_000_000,
                            "churn_share":.15,"manageable_share":.5,"at_risk_revenue":3_150_000_000,
                            "value_at_10pct":315_000_000,"value_at_15pct":472_500_000}}


def public_experiment():
    path=ROOT/"data/public/iranian_churn.csv"
    if not path.exists(): raise FileNotFoundError("The attributed UCI public CSV must be present.")
    meta=json.loads(path.with_suffix('.source.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta["sha256"]
    df=pd.read_csv(path); df.columns=[" ".join(c.split()) for c in df.columns]
    # Equal feature vectors cannot cross splits: conservative duplicate-group split.
    cols=[c for c in df.columns if c not in ["Churn","Status","Customer Value","Age","Age Group"]]
    key=pd.util.hash_pandas_object(df[cols],index=False)
    group=key.astype(str)
    unique=pd.DataFrame({"group":group,"label":df.Churn}).groupby("group",as_index=False).label.max()
    gtrain,gtest=train_test_split(unique.group,test_size=.25,random_state=SEED,stratify=unique.label)
    tr=group.isin(gtrain); te=group.isin(gtest)
    model=CatBoostClassifier(iterations=300,depth=4,learning_rate=.05,random_seed=SEED,
                             thread_count=4,verbose=False,allow_writing_files=False)
    model.fit(df.loc[tr,cols],df.loc[tr,"Churn"])
    probability=model.predict_proba(df.loc[te,cols])[:,1]
    # Deliberately simple process baseline: complaint, then low usage. No test labels.
    rule=(df.Complains*2 + (df["Seconds of Use"]<df.loc[tr,"Seconds of Use"].quantile(.25)).astype(int)).to_numpy()
    rates={k:float(df.loc[tr & (rule==k),"Churn"].mean()) for k in np.unique(rule[tr])}
    rp=np.array([rates.get(v,float(df.loc[tr,"Churn"].mean())) for v in rule[te]])
    y=df.loc[te,"Churn"].to_numpy()
    result={"source":meta,"rows":len(df),"train":int(tr.sum()),"test":int(te.sum()),
            "unique_feature_groups":len(unique),"feature_columns":cols,
            "group_overlap":len(set(group[tr]) & set(group[te])),
            "excluded":["Status","Customer Value","Age","Age Group"],
            "ml":summary(y,probability,probability),"rules":summary(y,rule[te],rp),
            "precision_difference_ci95":boot_precision(y,probability,rule[te],groups=group[te].to_numpy()),
            "ci_method":"Paired cluster bootstrap by identical feature vectors, 400 resamples",
            "split":"Group-disjoint stratified holdout, not a calendar out-of-time test.",
            "limitations":"Telecom in Iran, 3-month target, no action logs, no raw dates. Not RKO value or uplift evidence."}
    examples=df.loc[te,cols+["Churn"]].copy()
    examples["probability"]=probability
    examples["record"]=examples.index
    examples.to_csv(OUT/"public_predictions.csv",index=False)
    model.save_model(str(ROOT/"models/public.cbm"))
    return result


def safety_checks(df, test, risk, uplift, model, campaign):
    checks=[]
    def add(case, title, passed, evidence):
        checks.append({"case":case,"title":title,"passed":bool(passed),"evidence":evidence})
    banned=df.compliance | df.liquidation | df.tochka
    add(1,"Фильтр 115-ФЗ до модели",not test.compliance.any() and not set(campaign.client_id)&set(df.loc[df.compliance,"client_id"]),
        {"flagged_total":int(df.compliance.sum()),"scored_flagged":int(test.compliance.sum())})
    # Paired seasonal counterfactual: underlying business stays stable.
    pair=test.iloc[[0,0]].copy()
    pair["season_adjusted_ratio"]=1.; pair["yoy_ratio"]=1.; pair["days_no_incoming"]=0
    pair["turnover_ratio"]=[1.,.5]
    p=model.predict(pair)["any"]; delta=float(p[1]-p[0])
    add(2,"Сезонное падение не повышает скор",delta<=1e-8,{"paired_scores":p,"increase":delta,"tolerance":1e-8,
        "note":"Strict Gate 2 requirement; a small increase is still reported as a failure."})
    dates_ok=bool((df.feature_max_date<df.cutoff).all())
    embargo=bool(df.loc[df.split.eq("train"),"label_available"].max()<df.loc[df.split.eq("calibration"),"cutoff"].min()
                 and df.loc[df.split.eq("calibration"),"label_available"].max()<df.loc[df.split.eq("test"),"cutoff"].min())
    forbidden={"y_close","y_quiet","y_any","close_days","compliance","arm","retained","client_id"}
    # Mutation test: changed future labels cannot alter predictions on unchanged features.
    mutated=test.iloc[:30].copy(); mutated["y_close"]=1-mutated.y_close
    mutated["y_quiet"]=1-mutated.y_quiet; mutated["retained"]=1-mutated.retained
    invariant=np.array_equal(model.predict(mutated)["any"],risk["any"][:30])
    add(3,"Срез и зрелость таргета",dates_ok and embargo and not forbidden&set(FEATURES) and invariant,
        {"feature_dates_before_cutoff":dates_ok,"embargo":embargo,"future_mutation_invariant":invariant,
         "note":"Column lineage and mutation tests; not a complete real-DWH leakage audit."})
    chosen=np.zeros(len(test),bool); chosen[top(risk["any"])]=True
    groups={}
    for name,sub in test.groupby("industry"):
        loc=test.index.get_indexer(sub.index); y=sub.y_any.to_numpy()
        groups[name]={"n":len(sub),"positive":int(y.sum()),"recall":float(y[chosen[loc]].sum()/y.sum()),
                      "brier":float(brier_score_loss(y,risk["any"][loc]))}
    gap=max(v["recall"] for v in groups.values())-min(v["recall"] for v in groups.values())
    regions={}
    for name,sub in test.groupby("region"):
        loc=test.index.get_indexer(sub.index); y=sub.y_any.to_numpy()
        regions[str(name)]={"n":len(sub),"positive":int(y.sum()),"recall":float(y[chosen[loc]].sum()/y.sum()),
                            "brier":float(brier_score_loss(y,risk["any"][loc]))}
    region_gap=max(v["recall"] for v in regions.values())-min(v["recall"] for v in regions.values())
    add(4,"Разрыв полноты по отраслям и регионам ≤5 п.п.",max(gap,region_gap)<=.05,
        {"groups":groups,"recall_gap":gap,"regions":regions,"region_recall_gap":region_gap})
    allquotes=[quote(t,p,o) for t in ["advanced","professional"] for p in [0,5,10,11,40,100,180,300] for o in [1,2,3]]
    allowed=[q for q in allquotes if q["allowed"]]
    add(5,"Оффер не увеличивает текущую плату",all(q["after_year"]<=q["before_year"] and q["after_month"]<=q["current_month"] for q in allowed),
        {"grid":len(allquotes),"rejected":len(allquotes)-len(allowed)})
    # Independent literal expected values, including a bad package and a lower tariff trap.
    expected=[quote("advanced",50,1)["saving_year"]==5010,
              quote("advanced",50,1)["budget_cost"]==2970,
              quote("advanced",50,3)["saving_year"]==4060,
              quote("advanced",180,2)["allowed"] is False,
              monthly_cost("advanced",10)==1990]
    add(6,"Детерминированный расчёт экономии",all(expected),{"golden_cases":len(expected),"passed":sum(expected)})
    recent=set(df.loc[df.days_last_offer.lt(30),"client_id"])
    add(7,"Не более одного оффера за 30 дней",campaign.client_id.is_unique and not set(campaign.client_id)&recent,
        {"duplicates":int(campaign.client_id.duplicated().sum()),"recent_in_campaign":len(set(campaign.client_id)&recent)})
    add(8,"Переход в Точку исключён",not set(campaign.client_id)&set(df.loc[df.tochka,"client_id"]),
        {"excluded_known_migrations":int(df.tochka.sum()),"note":"Scenario business rule from Gate 2; current group ownership is not asserted."})
    drift=psi(test.fee_burden,test.fee_burden+1.5)
    blocked=False
    try: select_campaign(pd.DataFrame(),drift_psi=drift)
    except ValueError: blocked=True
    add(9,"Сдвиг распределения → сигнал и остановка",drift>.2 and blocked,{"psi":drift,"threshold":.2,"campaign_blocked":blocked,
        "note":"Detection and blocking are exercised; retraining/revalidation remains a human workflow."})
    return checks


def research_examples():
    templates={
      "price":["После роста платы за тариф перевёл обороты в другой банк.","Комиссии за платежи стали выше ожидаемого, сменил основной счёт.","Пакет не подходит моему числу переводов; ушёл из-за переплаты.","При низких оборотах обслуживание оказалось дорогим, сменил банк."],
      "forced":["Закрыл бизнес, расчётный счёт больше не нужен.","Банк ограничил операции; счёт закрыт по его инициативе.","Компания ликвидирована, обслуживание прекращено.","Документы по проверке не прошли, обслуживание остановлено."],
      "other":["Кредитор потребовал счёт в другом банке, туда ушли обороты.","Ключевой заказчик платит через другой банк, перенёс туда расчёты.","Не устроила работа сервиса, тарифная скидка вопрос не решит."],
      "stay":["Использую бухгалтерию и эквайринг, менять основной банк не собираюсь.","Плата устраивает, обороты сезонные, счёт оставляю основным."]}
    categories=["price"]*100+["forced"]*50+["other"]*30+["stay"]*20
    rng=np.random.default_rng(SEED); rng.shuffle(categories)
    records=[]
    for i,cat in enumerate(categories):
        text=templates[cat][i%len(templates[cat])]
        records.append({"id":f"EDU-{i+1:03d}","bank":["Т-Бизнес","СберБизнес","Альфа-Бизнес"][i%3],
                        "text":text,"reason":cat,"churn":cat!="stay","manageable":cat=="price"})
    pd.DataFrame(records).to_csv(ROOT/"data/reviews_200.csv",index=False)
    result={"n":200,"churn":180,"manageable":100,"share_among_churn":100/180,
            "gate2_threshold_decision_in_scenario":"Go", "human_interviews":0,
            "scenario_interviews":5,"human_double_annotation_kappa":None,
            "unique_texts":len(set(r["text"] for r in records)),
            "use":"Exercise coding guide and decision rules, not estimate population reason shares."}
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--clients",type=int,default=50_000)
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True); (ROOT/"models").mkdir(exist_ok=True)
    start=time.perf_counter()
    print("Generating 24-month panel",flush=True)
    df,truth=generate(args.clients,destination=ROOT/"data")
    # Stable pseudo-random order prevents deterministic client ID order favoring a rule.
    df=df.sample(frac=1,random_state=SEED).reset_index(drop=True)
    truth=truth.set_index("client_id").loc[df.client_id].reset_index()
    eligible=eligibility(df)
    parts={s:df.loc[eligible & df.split.eq(s)].copy() for s in ["train","calibration","test"]}
    train,cal,test=[parts[s] for s in ["train","calibration","test"]]
    tt=truth.set_index("client_id").loc[test.client_id].reset_index()
    print("Fitting two risk models",len(train),len(cal),len(test),flush=True)
    fitted=time.perf_counter(); model=ChurnModels().fit(train,cal)
    risk_seconds=time.perf_counter()-fitted
    risk_result,risk=evaluate_risk(test,model)
    print("Fitting four randomized-arm outcome models",flush=True)
    fitted=time.perf_counter(); uplift_model=UpliftModels().fit(train)
    uplift_seconds=time.perf_counter()-fitted
    uplift_result,uplift=evaluate_uplift(test,tt,uplift_model)
    print("Building safe recommendations and campaign",flush=True)
    recommendations=recommend(test,uplift)
    campaign=select_campaign(recommendations)
    campaign.to_csv(OUT/"campaign.csv",index=False)
    risk_prediction=test.copy()
    for k,v in risk.items(): risk_prediction[f"risk_{k}"]=v
    for k in [1,2,3]: risk_prediction[f"uplift_{k}"]=uplift[:,k-1]
    risk_prediction=risk_prediction.merge(recommendations,on="client_id",how="left")
    risk_prediction.to_csv(OUT/"predictions.csv.gz",index=False)
    comparison=policy_comparison(test,tt,risk,uplift)
    safety=safety_checks(df,test,risk,uplift,model,campaign)
    # SHAP for a bounded review sample. Explanations are for raw quiet-risk log odds.
    review=test.iloc[top(risk["any"])[:30]]
    shap=model.explain(review)
    shap_rows=[]
    for j,row in enumerate(review.itertuples()):
        order=np.argsort(-np.abs(shap[j,:-1]))[:3]
        shap_rows.append({"client_id":row.client_id,"contributions":[{"feature":FEATURES[k],"log_odds":float(shap[j,k])} for k in order]})
    dump("shap_examples.json",shap_rows)
    print("Benchmark and public-data check",flush=True)
    lat={"single_two_risks":timed(lambda:model.predict(test.iloc[:1]),20),
         "batch_all_models":timed(lambda:(model.predict(test),uplift_model.predict(test)),7),
         "sample_n":len(test),"thread_count":4,"training_risk_seconds":risk_seconds,
         "training_uplift_seconds":uplift_seconds,"machine":platform.machine(),
         "system":platform.system(),"python":sys.version.split()[0]}
    # Actual 1M-row inference benchmark, 10 bounded chunks: includes DataFrame work.
    chunk=pd.concat([test]*int(np.ceil(100_000/len(test))),ignore_index=True).iloc[:100_000]
    bench=time.perf_counter()
    for _ in range(10): model.predict(chunk); uplift_model.predict(chunk)
    lat["actual_1m_all_models_seconds"]=time.perf_counter()-bench
    lat["cost_inputs"]={"cpu_rub_hour":20,"hours_month":730,"pricing":"planning assumption, not a provider quote"}
    lat["compute_rub_per_prediction"]=lat["actual_1m_all_models_seconds"]/3600*20/1_000_000
    lat["always_on_cpu_month_rub"]=730*20
    lat["daily_batch_cpu_month_rub"]=lat["actual_1m_all_models_seconds"]/3600*20*30
    pub=public_experiment()
    false_positive=np.flatnonzero((test.y_any.to_numpy()==0)&np.isin(np.arange(len(test)),top(risk["any"])))
    error_ix=int(false_positive[np.argmax(risk["any"][false_positive])])
    error=test.iloc[error_ix]
    example={"client_id":error.client_id,"predicted_risk":float(risk["any"][error_ix]),
             "observed_churn":int(error.y_any),"type":"False positive in the selected top 10%",
             "explanation":"A high risk is a probability, not an observed departure. One realized non-event does not prove a probability is miscalibrated."}
    split_stats={}
    for name,part in parts.items():
        split_stats[name]={"n":len(part),"cutoff_min":str(part.cutoff.min().date()),
                           "cutoff_max":str(part.cutoff.max().date()),"label_available_max":str(part.label_available.max().date())}
    bundle={"risk":model,"uplift":uplift_model}
    with (ROOT/"models/bundle.pkl").open("wb") as f: pickle.dump(bundle,f)
    versions={p:importlib.metadata.version(p) for p in ["numpy","pandas","catboost","scikit-learn","scipy","scikit-uplift"]}
    manifest={"seed":SEED,"clients":args.clients,"months":24,"panel_rows":args.clients*24,
              "eligible":int(eligible.sum()),"splits":split_stats,"features":FEATURES,"versions":versions,
              "research":research_examples(),"model_bytes":(ROOT/"models/bundle.pkl").stat().st_size,
              "elapsed_seconds":time.perf_counter()-start,"data_kind":"Educational simulation + separate UCI real telecom benchmark",
              "test_used_for_tuning":False}
    dump("manifest.json",manifest); dump("risk_metrics.json",risk_result)
    dump("uplift_metrics.json",uplift_result); dump("policies.json",comparison)
    dump("safety.json",safety); dump("economics.json",economics()); dump("latency.json",lat)
    dump("public_metrics.json",pub); dump("demo_error.json",example)
    from refresh_notebook import refresh
    refresh()
    print(json.dumps({"risk_any":risk_result["any"],"uplift":uplift_result,
                      "safety_passed":sum(s["passed"] for s in safety),"campaign":len(campaign),
                      "seconds":manifest["elapsed_seconds"]},ensure_ascii=False,default=str),flush=True)


if __name__=="__main__": main()
