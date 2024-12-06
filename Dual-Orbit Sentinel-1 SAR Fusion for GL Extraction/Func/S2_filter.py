import ee
from functools import partial
def clip_AOI(col, AOI): return col.clip(AOI)
def add_cloud_bands(img,CLD_PRB_THRESH):
    """Define a function to add the s2cloudless probability layer
    and derived cloud mask as bands to an S2 SR image input."""
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img,NIR_DRK_THRESH,CLD_PRJ_DIST):
    not_water = img.select('SCL').neq(6)

    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img,BUFFER,CLD_PRJ_DIST,NIR_DRK_THRESH,CLD_PRB_THRESH):

    img_cloud = add_cloud_bands(img,CLD_PRB_THRESH)

    img_cloud_shadow = add_shadow_bands(img_cloud,NIR_DRK_THRESH=NIR_DRK_THRESH,CLD_PRJ_DIST=CLD_PRJ_DIST)

    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    return img_cloud_shadow.addBands(is_cld_shdw)

def apply_cld_shdw_mask(img):
    not_cld_shdw = img.select('cloudmask').Not()
    return img.select(['B.*','clouds','dark_pixels','shadows','cloudmask']).updateMask(not_cld_shdw)

def merge_s2_collection(aoi, start_date, end_date,CLOUD_FILTER,BUFFER,CLD_PRJ_DIST,CLD_PRB_THRESH,NIR_DRK_THRESH):
    s2_sr_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)).map(partial(clip_AOI,AOI=aoi)))

    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date).map(partial(clip_AOI,AOI=aoi)))

    s2_sr_cld_col = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'})
    }))

    s2_sr_cld_col_disp = s2_sr_cld_col.map(partial(add_cld_shdw_mask,BUFFER=BUFFER,
                            CLD_PRJ_DIST=CLD_PRJ_DIST,
                            CLD_PRB_THRESH=CLD_PRB_THRESH,
                            NIR_DRK_THRESH=NIR_DRK_THRESH))
    s2_sr_median = s2_sr_cld_col_disp.map(apply_cld_shdw_mask).median().clip(aoi).int16()
    return s2_sr_median