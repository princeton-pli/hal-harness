from datasets import load_dataset
swebench = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')


print(swebench.to_list()[0])