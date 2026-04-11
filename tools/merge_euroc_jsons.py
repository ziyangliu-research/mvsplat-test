import argparse
import json
from pathlib import Path

# 合并训练json脚本

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    merged = {}
    for p in args.inputs:
        path = Path(p)
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        overlap = set(merged.keys()) & set(data.keys())
        if overlap:
            raise RuntimeError(f'Duplicate keys found when merging {path}: {sorted(list(overlap))[:5]}')
        merged.update(data)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f'Saved merged json: {out} ({len(merged)} items)')


if __name__ == '__main__':
    main()
