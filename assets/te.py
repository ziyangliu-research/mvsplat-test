import json, os
scene_name = 'rgbd_bonn_static'
num_pairs = 500
step = 10
offset = 5  # midpoint target
index = {}
for k in range(num_pairs):
    left = k * step
    right = left + step
    target = left + offset
    key = f'{scene_name}_{k:04d}'
    index[key] = {
        'context': [left, right],
        'target': [target]
    }
path = '/workspace/evaluation_index_rgbd_bonn_static_nctx2_step200_mid.json'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(index, f, ensure_ascii=False, indent=2)
print(path)
print(list(index.items())[:3])
print(list(index.items())[-3:])