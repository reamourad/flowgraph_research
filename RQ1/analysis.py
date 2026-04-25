import json, pandas as pd

results = [json.loads(l) for l in open("results_multimodel/results_gpt-oss-120b.jsonl")]
df = pd.json_normalize(results)  # flattens metrics.* into columns
print(df.groupby("approach")[["metrics.passed", "metrics.code_bleu", 
                               "metrics.cc_delta"]].mean())