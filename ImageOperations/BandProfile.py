sent_NDVI = "(B8-B4)/(B8+B4)"
sent_NDVI2 = "(B8a-B4)/(B8a+B4)"
sent_bands_dict = {
    "B1": 0,
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B5": 4,
    "B6": 5,
    "B7": 6,
    "B8": 7,
    "B8a": 8,
    "B9": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}
rgbnir_bands_dict = {
    "B2": 0,
    "B3": 1,
    "B4": 2,
    "B8": 3,
}

def update_band_profile(band_dict, image):
    band_profile = band_dict.copy()
    for k in band_profile.keys():
        band_profile.update({k:x[band_profile[k]]})
        return band_profile
