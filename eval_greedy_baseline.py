"""
贪心基线评测 — 重构版（支持 BGE / sentence-transformers）

移除 DPR 硬编码，改为通用 DenseGreedy，支持任意编码器。

运行示例：
    # BM25 baseline
    python eval_greedy_baseline.py --baseline bm25 --device cuda

    # BGE greedy baseline（替代旧 DPR）
    python eval_greedy_baseline.py --baseline dense \
        --encoder_model BAAI/bge-base-en-v1.5 --device cuda

    # 两者对比
    python eval_greedy_baseline.py --baseline both \
        --encoder_model BAAI/bge-base-en-v1.5 --device cuda
"""
import argparse, json, logging, os, re, string, sys
from collections import defaultdict
from typing import Dict, List, Optional, Set
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Text utils ────────────────────────────────────────────────────────────────

def _norm(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(s.split())

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

def compute_metrics(sel_docs, sel_titles, pred_ans, gold_ans, gold_titles, gold_facts):
    a_em = ans_em(pred_ans, gold_ans)
    a_f1 = ans_f1(pred_ans, gold_ans)
    pf   = sf_sents(sel_docs, pred_ans)
    s_em = sup_em(pf, gold_facts)
    s_f1 = sup_f1(pf, gold_facts)
    return {
        "ans_em":       a_em,  "ans_f1":       a_f1,
        "sup_em":       s_em,  "sup_f1":       s_f1,
        "joint_em":     float(a_em==1.0 and s_em==1.0),
        "joint_f1":     a_f1 * s_f1,
        "joint_recall": float(gold_titles.issubset(set(sel_titles))),
        "sf_hit_hop1":  float(sel_titles[0] in gold_titles) if sel_titles else 0.0,
        "sf_hit_hop2":  float(sel_titles[1] in gold_titles) if len(sel_titles)>=2 else 0.0,
    }


# ── BM25 Greedy ───────────────────────────────────────────────────────────────

def bm25_score(qtoks, text, k1=1.5, b=0.75, avg_dl=100):
    dt = text.lower().split(); dl = len(dt)
    tf = defaultdict(int)
    for t in dt: tf[t] += 1
    return sum(tf[t]*(k1+1)/(tf[t]+k1*(1-b+b*dl/avg_dl)) for t in qtoks if t in tf)

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
    total = len(data)
    logger.info(f"BM25 greedy evaluation: {total} samples")
    for idx, item in enumerate(data):
        if (idx+1) % 500 == 0 or idx == 0:
            logger.info(f"BM25 progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)")
        pool        = [{"title": t, "text": " ".join(s)} for t, s in item.get("context", [])]
        gold_titles = set(sf[0] for sf in item["supporting_facts"])
        gold_facts  = [(sf[0], sf[1]) for sf in item["supporting_facts"]]
        sel_docs, sel_titles = [], []
        for hop in range(max_hops):
            query = item["question"] + (" " + " ".join(sel_titles) if sel_titles else "")
            doc = bm25_select(query, pool, set(sel_titles))
            if doc is None: break
            sel_docs.append(doc); sel_titles.append(doc["title"])
        predicted = reader.predict(item["question"], sel_docs) if sel_docs else ""
        for k, v in compute_metrics(sel_docs, sel_titles, predicted,
                                    item["answer"], gold_titles, gold_facts).items():
            results[k].append(v)
    return {k: float(np.mean(v)) for k, v in results.items()}


# ── Dense Greedy（通用：BGE / e5 / MiniLM 等）────────────────────────────────

class DenseGreedy:
    """
    用任意 EncoderBackend 做贪心检索。
    每跳计算 query 和候选文档的余弦相似度，选最高分文档。
    """
    def __init__(self, encoder_model="BAAI/bge-base-en-v1.5", backend_type="auto",
                 device="cuda", cache_dir="cache",
                 query_prefix="", doc_prefix="", batch_size=64):
        from retriever import create_encoder_backend
        logger.info(f"Loading encoder for Dense greedy: {encoder_model}")
        self.backend = create_encoder_backend(
            model_name=encoder_model, backend_type=backend_type,
            device=device, cache_dir=cache_dir,
            query_prefix=query_prefix, doc_prefix=doc_prefix,
        )
        self.batch_size = batch_size
        logger.info(f"Dense greedy ready: dim={self.backend.dim}")

    def select(self, query: str, pool: List[Dict], exclude: Set[str]) -> Optional[Dict]:
        avail = [d for d in pool if d["title"] not in exclude]
        if not avail: return None
        # 编码 query
        q_emb = (self.backend.encode_query([query])
                 if hasattr(self.backend, "encode_query")
                 else self.backend.encode([query]))
        # 编码候选文档
        doc_texts = [f"{d['title']}. {d['text'][:400]}" for d in avail]
        d_emb = (self.backend.encode_docs(doc_texts, batch_size=self.batch_size)
                 if hasattr(self.backend, "encode_docs")
                 else self.backend.encode(doc_texts, batch_size=self.batch_size))
        # 内积相似度（已 L2 归一化 = 余弦相似度）
        scores = (q_emb @ d_emb.T).squeeze(0)
        return avail[int(scores.argmax())]


def evaluate_dense_greedy(data, reader, dense: DenseGreedy, max_hops=2):
    results = defaultdict(list)
    total = len(data)
    logger.info(f"Dense greedy evaluation: {total} samples")
    for idx, item in enumerate(data):
        if (idx+1) % 200 == 0 or idx == 0:
            logger.info(f"Dense progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)")
        pool        = [{"title": t, "text": " ".join(s)} for t, s in item.get("context", [])]
        gold_titles = set(sf[0] for sf in item["supporting_facts"])
        gold_facts  = [(sf[0], sf[1]) for sf in item["supporting_facts"]]
        sel_docs, sel_titles = [], []
        for hop in range(max_hops):
            query = item["question"] + (" " + " ".join(sel_titles) if sel_titles else "")
            doc = dense.select(query, pool, set(sel_titles))
            if doc is None: break
            sel_docs.append(doc); sel_titles.append(doc["title"])
        predicted = reader.predict(item["question"], sel_docs) if sel_docs else ""
        for k, v in compute_metrics(sel_docs, sel_titles, predicted,
                                    item["answer"], gold_titles, gold_facts).items():
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

def print_comparison(bm, dm, dense_label="Dense Greedy"):
    print(f"\n{'='*72}\n  {f'BM25 vs {dense_label}':^68}\n{'='*72}")
    print(f"  {'':24} {'BM25 EM':>8} {'BM25 F1':>8}  {'Dense EM':>8} {'Dense F1':>8}")
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
    p = argparse.ArgumentParser(description="Greedy Baseline Evaluation (BM25 / Dense)")
    p.add_argument("--data_dir",          default="data/hotpotqa")
    p.add_argument("--reader_model",      default="deepset/roberta-base-squad2")
    p.add_argument("--device",            default="cuda")
    p.add_argument("--cache_dir",         default="cache")
    p.add_argument("--max_samples",       type=int, default=None)
    p.add_argument("--max_hops",          type=int, default=2)
    p.add_argument("--output_dir",        default="outputs")
    p.add_argument("--baseline",          default="both",
                   choices=["bm25","dense","both"])
    p.add_argument("--encoder_model",     default="BAAI/bge-base-en-v1.5")
    p.add_argument("--encoder_backend",   default="auto",
                   choices=["auto","sentence_transformer","huggingface"])
    p.add_argument("--query_prefix",      default="")
    p.add_argument("--doc_prefix",        default="")
    p.add_argument("--encode_batch_size", type=int, default=64)
    args = p.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(args.output_dir, exist_ok=True)

    dev_path = os.path.join(args.data_dir, "hotpot_dev_distractor.json")
    with open(dev_path) as f:
        dev = json.load(f)
    import random; random.seed(42)
    data = random.sample(dev, args.max_samples) if args.max_samples else dev
    logger.info(f"Evaluating on {len(data)} samples")

    from reader import ExtractiveReader
    reader = ExtractiveReader(model_name=args.reader_model, device=args.device)

    results = {}
    dense_label = f"Dense ({args.encoder_model.split('/')[-1]})"

    if args.baseline in ("bm25", "both"):
        logger.info("=== BM25 Greedy ===")
        bm = evaluate_bm25_greedy(data, reader, args.max_hops)
        print_table(bm, "BM25 Greedy Baseline (Context Oracle)")
        results["bm25"] = bm
        json.dump(bm, open(f"{args.output_dir}/bm25_baseline.json","w"), indent=2)

    if args.baseline in ("dense", "both"):
        logger.info(f"=== Dense Greedy ({args.encoder_model}) ===")
        dense = DenseGreedy(
            encoder_model=args.encoder_model,
            backend_type=args.encoder_backend,
            device=args.device, cache_dir=args.cache_dir,
            query_prefix=args.query_prefix, doc_prefix=args.doc_prefix,
            batch_size=args.encode_batch_size,
        )
        dm = evaluate_dense_greedy(data, reader, dense, args.max_hops)
        print_table(dm, f"Dense Greedy — {args.encoder_model.split('/')[-1]}")
        results["dense"] = dm
        safe_name = args.encoder_model.replace("/", "_")
        json.dump(dm, open(f"{args.output_dir}/dense_baseline_{safe_name}.json","w"), indent=2)

    if args.baseline == "both":
        print_comparison(results["bm25"], results["dense"], dense_label)
        json.dump(results, open(f"{args.output_dir}/baseline_comparison.json","w"), indent=2)

if __name__ == "__main__":
    main()