import os


base_dir = '/mnt/Data/mrt/RealWorldOccludedFaces/images'
masked_dir = f"{base_dir}/masked"
neutral_dir = f"{base_dir}/neutral"
sunglasses_dir = f"{base_dir}/sunglasses"

masked_items = os.listdir(masked_dir)
# for item in masked_items:
#     print(item.split('_wearing_mask')[0])
neutral_items = os.listdir(neutral_dir)
# for item in neutral_items:
#     print(item)
sunglasses_items = os.listdir(sunglasses_dir)
# for item in sunglasses_items:
#     print(item.split('_wearing_sunglasses')[0])

for item in masked_items:
    if not os.listdir(f"{masked_dir}/{item}"):
        print(f'Masked folder ->\t{item} has no images')
for item in neutral_items:
    if not os.listdir(f"{neutral_dir}/{item}"):
        print(f'Neutral folder ->\t{item} has no images')
for item in sunglasses_items:
    if not os.listdir(f"{sunglasses_dir}/{item}"):
        print(f'Sunglasses folder ->\t{item} has no images')

# TODO: Remove directories not containing any image
# Masked folder ->	mark_warner_wearing_mask has no images
# Masked folder ->	kiki_bertens_wearing_mask has no images
# Masked folder ->	tammy_baldwin_wearing_mask has no images
# Masked folder ->	karim_benzema_wearing_mask has no images
# Masked folder ->	victoria_azarenka_wearing_mask has no images
# Masked folder ->	hasan_salihamidzic_wearing_mask has no images
# Masked folder ->	ilhan_omar_wearing_mask has no images
# Sunglasses folder ->	patrick_leahy_wearing_sunglasses has no images
# Sunglasses folder ->	joe_manchin_wearing_sunglasses has no images
# Sunglasses folder ->	amy_klobuchar_wearing_sunglasses has no images
# Sunglasses folder ->	mark_warner_wearing_sunglasses has no images
# Sunglasses folder ->	bernie_sanders_wearing_sunglasses has no images
# Sunglasses folder ->	bernard_arnault_wearing_sunglasses has no images
# Sunglasses folder ->	dick_durbin_wearing_sunglasses has no images
# Sunglasses folder ->	nicole_kidman_wearing_sunglasses has no images
# Sunglasses folder ->	patty_murray_wearing_sunglasses has no images
# Sunglasses folder ->	kiki_bertens_wearing_sunglasses has no images
# Sunglasses folder ->	daniil_medvedev_wearing_sunglasses has no images
# Sunglasses folder ->	tammy_baldwin_wearing_sunglasses has no images
# Sunglasses folder ->	jean_castex_wearing_sunglasses has no images
# Sunglasses folder ->	dominic_thiem_wearing_sunglasses has no images
# Sunglasses folder ->	simona_halep_wearing_sunglasses has no images
# Sunglasses folder ->	kamala_haris_wearing_sunglasses has no images
# Sunglasses folder ->	donald_trump_wearing_sunglasses has no images
# Sunglasses folder ->	ilhan_omar_wearing_sunglasses has no images
# Sunglasses folder ->	andrey_rublev_wearing_sunglasses has no images
# Sunglasses folder ->	iga_swiatek_wearing_sunglasses has no images
# Sunglasses folder ->	diego_schwartzman_wearing_sunglasses has no images
# Sunglasses folder ->	cory_booker_wearing_sunglasses has no images
# Sunglasses folder ->	fahrettin_koca_wearing_sunglasses has no images
# Sunglasses folder ->	debbie_stabenow_wearing_sunglasses has no images
