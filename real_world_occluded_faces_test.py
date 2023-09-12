import os


base_dir = '/mnt/Data/mrt/RealWorldOccludedFaces/images'
masked_dir = f"{base_dir}/masked"
neutral_dir = f"{base_dir}/neutral"
sunglasses_dir = f"{base_dir}/sunglasses"

masked_items = os.listdir(masked_dir)
neutral_items = os.listdir(neutral_dir)
sunglasses_items = os.listdir(sunglasses_dir)

for item in masked_items:
    if not os.listdir(f"{masked_dir}/{item}"):
        print(f'Masked folder ->\t{item} has no images, removing directory')
        os.rmdir(f'{masked_dir}/{item}')
for item in neutral_items:
    if not os.listdir(f"{neutral_dir}/{item}"):
        print(f'Neutral folder ->\t{item} has no images, removing directory')
        os.rmdir(f'{neutral_dir}/{item}')
for item in sunglasses_items:
    if not os.listdir(f"{sunglasses_dir}/{item}"):
        print(f'Sunglasses folder ->\t{item} has no images, removing directory')
        os.rmdir(f'{sunglasses_dir}/{item}')
