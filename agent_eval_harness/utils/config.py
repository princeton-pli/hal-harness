import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add SWE-bench specific configurations
    config.setdefault('swe_bench_dataset', 'princeton-nlp/SWE-bench_Lite')
    config.setdefault('swe_bench_max_workers', 1)
    
    return config