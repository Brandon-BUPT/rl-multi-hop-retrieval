"""
贪心基线评测 (BM25 / DPR) on Context Oracle Candidates

运行：
    python eval_greedy_baseline.py --baseline both --device cuda:1
"""
import argparse, json, logging, os, re, string, sys
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Text utils ────────────────────────────────────────────────────────────────

def _norm(s):
    def ra(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def rp(t): return "".join(c for c in t if c not in set(string.punctuation))
    def ws(t): return " ".join(t.split())
    return ws(ra(rp(s.lower())))

def _toks(s): return _norm(s).split() if s else []
def ans_em(p, g): return float(_norm(p) == _norm(g))
def ans_f1(p, g):
    pt, gt = _toks(p), _toks(g)
    cnt = defaultdict(int)
    for t in pt: cnt[t] += 1
    ov = 0
    for t in gt:
        if cnt[t] > 0: ov += 1; cnt[t] -= 1
    if not pt or not gt: return float(pt == gt)
    pr, r = ov/len(pt), ov/len(gt)
    return 2*pr*r/(pr+r) if pr+r else 0.0
def sup_em(pf, gf): return float(set(pf) == set(gf))
def sup_f1(pf, gf):
    ps, gs = set(pf), set(gf)
    tp = len(ps & gs)
    if not ps or not gs: return float(ps == gs)
    pr, r = tp/len(ps), tp/len(gs)
    return 2*pr*r/(pr+r) if pr+r else 0.0
def sf_sents(docs, ans):
    at = set(_toks(ans))
    pred = []
    for doc in docs:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc["text"].strip()) if s.strip()]
        if not sents: continue
        bi, bs = 0, -1.0
        for i, s in enumerate(sents):
            sc = len(at & set(_toks(s))) / (len(set(_toks(s))) + 1e-9)
            if sc > bs: bs, bi = sc, i
        pred.append((doc["title"], bi))
    return pred

def metrics(sel_docs, sel_titles, pred_ans, gold_ans, gold_titles, gold_facts):
    a_em = ans_em(pred_ans, gold_ans); a_f1 = ans_f1(pred_ans, gold_ans)
    pf = sf_sents(sel_docs, pred_ans)
    s_em = sup_em(pf, gold_facts); s_f1 = sup_f1(pf, gold_facts)
    return {
        "ans_em": a_em, "ans_f1": a_f1, "sup_em": s_em, "sup_f1": s_f1,
        "joint_em": float(a_em==1.0 and s_em==1.0), "joint_f1": a_f1*s_f1,
        "joint_recall": float(gold_titles.issubset(set(sel_titles))),
        "sf_hit_hop1": float(sel_titles[0] in gold_titles) if sel_titles else 0.0,
        "sf_hit_hop2": float(sel_titles[1] in gold_titles) if len(sel_titles)>=2 else 0.0,
    }


# ── BM25 ──────────────────────────────────────────────────────────────────────

def bm25_score(qtoks, text, k1=1.5, b=0.75):
    dt = text.lower().split(); dl = len(dt)
    tf = defaultdict(int)
    for t in dt: tf[t] += 1
    return sum(tf[t]*(k1+1)/(tf[t]+k1*(1-b+b*dl/100)) for t in qtoks if t in tf)

def bm25_select(query, pool, exclude):
    qt = query.lower().split()
    best, bs = None, -1.0
    for d in pool:
        if d["title"] in exclude: continue
        s = bm25_score(qt, d["title"]+" "+d["text"])
        if s > bs: bs, best = s, d
    return best

def evaluate_bm25_greedy(data, reader, max_hops=2):
    results = defaultdict(list)
    for item in data:
        pool = [{"title": t, "text": " ".join(s)} for t, s in item.get("context", [])]
        gt = set(sf[0] for sf in item["supporting_facts"])
        gf = [(sf[0], sf[1]) for sf in item["supporting_facts"]]
        sel_docs, sel_titles = [], []
        for hop in range(max_hops):
            q = item["question"] + (" " + " ".join(sel_titles) if sel_titles else "")
            d = bm25_select(q, pool, set(sel_titles))
            if d is None: break
            sel_docs.append(d); sel_titles.append(d["title"])
        pred = reader.predict(item["question"], sel_docs) if sel_docs else ""
        for k, v in metrics(sel_docs, sel_titles, pred, item["answer"], gt, gf).items():
            results[k].append(v)
    return {k: float(np.mean(v)) for k, v in results.items()}


# ── DPR ───────────────────────────────────────────────────────────────────────

class DPRGreedy:
    def __init__(self, device="cuda"):
        import torch
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
        self.device = device
        self.qt = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.qe = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device).eval()
        self.ct = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.ce = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device).eval()

    def select(self, query, pool, exclude):
        import torch, torch.nn.functional as F
        avail = [d for d in pool if d["title"] not in exclude]
        if not avail: return None
        qenc = self.qt(query, max_length=128, truncation=True, return_tensors="pt").to(self.device)
        texts = [f"{d['title']} {d['text'][:400]}" for d in avail]
        cenc = self.ct(texts, max_length=256, truncation=True, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            qvec = self.qe(**qenc).pooler_output
            cvec = self.ce(**cenc).pooler_output
        scores = F.cosine_similarity(qvec, cvec, dim=-1)
        return avail[scores.argmax().item()]

def evaluate_dpr_greedy(data, reader, dpr, max_hops=2):
    results = defaultdict(list)
    for i, item in enumerate(data):
        if i % 100 == 0: logger.info(f"DPR greedy {i}/{len(data)}")
        pool = [{"title": t, "text": " ".join(s)} for t, s in item.get("context", [])]
        gt = set(sf[0] for sf in item["supporting_facts"])
        gf = [(sf[0], sf[1]) for sf in item["supporting_facts"]]
        sel_docs, sel_titles = [], []
        for hop in range(max_hops):
            q = item["question"] + (" " + " ".join(sel_titles) if sel_titles else "")
            d = dpr.select(q, pool, set(sel_titles))
            if d is None: break
            sel_docs.append(d); sel_titles.append(d["title"])
        pred = reader.predict(item["question"], sel_docs) if sel_docs else ""
        for k, v in metrics(sel_docs, sel_titles, pred, item["answer"], gt, gf).items():
            results[k].append(v)
    return {k: float(np.mean(v)) for k, v in results.items()}


# ── Print ─────────────────────────────────────────────────────────────────────

def print_table(m, label="Greedy Baseline"):
    print(f"\n{'='*58}\n  {label:^54}\n{'='*58}")
    print(f"  {'':24} {'EM':>8}  {'F1':>8}\n  {'-'*54}")
    for key, name in [("ans","Answer"),("sup","Sup Facts"),("joint","Joint")]:
        print(f"  {name:24} {m[f'{key}_em']:>8.4f}  {m[f'{key}_f1']:>8.4f}")
    print(f"  {'-'*54}")
    print(f"  joint_recall:          {m['joint_recall']:.4f}")
    print(f"  sf_hit_hop1:           {m['sf_hit_hop1']:.4f}")
    print(f"  sf_hit_hop2:           {m['sf_hit_hop2']:.4f}")
    print(f"{'='*58}")

def print_comparison(bm, dm):
    print(f"\n{'='*72}\n  {'BM25 vs DPR Greedy Comparison':^68}\n{'='*72}")
    print(f"  {'':24} {'BM25 EM':>8} {'BM25 F1':>8}  {'DPR EM':>8} {'DPR F1':>8}")
    print(f"  {'-'*68}")
    for key, name in [("ans","Answer"),("sup","Sup Facts"),("joint","Joint")]:
        print(f"  {name:24} {bm[f'{key}_em']:>8.4f} {bm[f'{key}_f1']:>8.4f}  {dm[f'{key}_em']:>8.4f} {dm[f'{key}_f1']:>8.4f}")
    print(f"  {'-'*68}")
    print(f"  {'joint_recall':24} {bm['joint_recall']:>8.4f} {'':>8}  {dm['joint_recall']:>8.4f}")
    print(f"  {'sf_hit_hop1':24} {bm['sf_hit_hop1']:>8.4f} {'':>8}  {dm['sf_hit_hop1']:>8.4f}")
    print(f"  {'sf_hit_hop2':24} {bm['sf_hit_hop2']:>8.4f} {'':>8}  {dm['sf_hit_hop2']:>8.4f}")
    print(f"{'='*72}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     default="data/hotpotqa")
    parser.add_argument("--reader_model", default="deepset/roberta-base-squad2")
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--max_samples",  type=int, default=None, help="None=全量")
    parser.add_argument("--max_hops",     type=int, default=2)
    parser.add_argument("--baseline",     default="both", choices=["bm25","dpr","both"])
    parser.add_argument("--output_dir",   default="outputs")
    args = parser.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(args.data_dir, "hotpot_dev_distractor_v1.json")) as f:
        dev = json.load(f)
    import random; random.seed(42)
    data = random.sample(dev, args.max_samples) if args.max_samples else dev
    logger.info(f"Evaluating on {len(data)} samples")

    from reader import ExtractiveReader
    reader = ExtractiveReader(model_name=args.reader_model, device=args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    if args.baseline in ("bm25", "both"):
        logger.info("=== BM25 Greedy ===")
        bm = evaluate_bm25_greedy(data, reader, args.max_hops)
        print_table(bm, "BM25 Greedy Baseline (Context Oracle)")
        results["bm25"] = bm
        json.dump(bm, open(f"{args.output_dir}/bm25_baseline.json","w"), indent=2)

    if args.baseline in ("dpr", "both"):
        logger.info("=== DPR Greedy ===")
        dpr = DPRGreedy(device=args.device)
        dm = evaluate_dpr_greedy(data, reader, dpr, args.max_hops)
        print_table(dm, "DPR Greedy Baseline (Context Oracle)")
        results["dpr"] = dm
        json.dump(dm, open(f"{args.output_dir}/dpr_baseline.json","w"), indent=2)

    if args.baseline == "both":
        print_comparison(results["bm25"], results["dpr"])
        json.dump(results, open(f"{args.output_dir}/baseline_comparison.json","w"), indent=2)

if __name__ == "__main__":
    main()