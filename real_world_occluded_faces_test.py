import os


base_dir = '/mnt/Data/mrt/RealWorldOccludedFaces/images'
masked_dir = f"{base_dir}/masked"
neutral_dir = f"{base_dir}/neutral"
sunglasses_dir = f"{base_dir}/sunglasses"

items = os.listdir(masked_dir)
for item in items:
    print(item.split('_wearing_mask')[0])

items = os.listdir(neutral_dir)
for item in items:
    print(item)

items = os.listdir(sunglasses_dir)
for item in items:
    print(item.split('_wearing_sunglasses')[0])
