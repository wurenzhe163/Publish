import os
import ee
import geemap
from .Correct_filter import *
import copy
from tqdm import tqdm, trange
from .New_Correct import *
from .Correct_filter import volumetric_model_SCF
from osgeo import gdal

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created successfully')
    return path

def delList(L):
    """Remove duplicate elements"""
    return list(dict.fromkeys(L))

def Open_close(img, radius=10):
    '''Morphological opening and closing operation'''
    kernel = ee.Kernel.square(**{'radius': radius, 'units': 'meters'})
    min_img = img.reduceNeighborhood(**{'reducer': ee.Reducer.min(), 'kernel': kernel})
    opening = min_img.reduceNeighborhood(**{'reducer': ee.Reducer.max(), 'kernel': kernel})
    max_img = opening.reduceNeighborhood(**{'reducer': ee.Reducer.max(), 'kernel': kernel})
    closing = max_img.reduceNeighborhood(**{'reducer': ee.Reducer.min(), 'kernel': kernel})
    return closing

def calculate_iou(geometry1, geometry2):
    intersection = geometry1.intersection(geometry2)
    union = geometry1.union(geometry2)
    intersection_area = intersection.area()
    union_area = union.area()
    return intersection_area.divide(union_area)

# --------------------Normalization----------------------
def get_minmax(Image: ee.Image, scale: int = 10):
    '''Get min and max of an image band'''
    obj = Image.reduceRegion(reducer=ee.Reducer.minMax(), geometry=Image.geometry(), scale=scale, bestEffort=True)
    return obj.rename(**{'from': obj.keys(), 'to': ['max', 'min']})

def get_meanStd(Image: ee.Image, scale: int = 10):
    '''Get mean and standard deviation of an image band'''
    obj = Image.reduceRegion(reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                             geometry=Image.geometry(), scale=scale, bestEffort=True)
    return obj.rename(**{'from': obj.keys(), 'to': ['mean', 'std']})

def minmax_norm(Image: ee.Image, Bands, region, scale: int = 10, withbound=False):
    for i, band in enumerate(Bands):
        band_img = Image.select(band)
        if withbound:
            histogram = get_histogram(Image, region, scale).getInfo()
            bin_centers, counts = np.array(histogram['bucketMeans']), np.array(histogram['histogram'])
            bounds = HistBoundary(counts, y=100)
            min_val, max_val = bin_centers[bounds['indexFront']], bin_centers[bounds['indexBack']]
            Image = Image.where(Image.lt(min_val), min_val).where(Image.gt(max_val), max_val)
            nominator = band_img.subtract(min_val)
            denominator = max_val - min_val
        else:
            minmax = get_minmax(band_img, scale)
            nominator = band_img.subtract(minmax.get('min'))
            denominator = minmax.get('max').subtract(minmax.get('min'))
        
        if i == 0:
            result = nominator.divide(denominator)
        else:
            result = result.addBands(nominator.divide(denominator))
    return result

def meanStd_norm(Image: ee.Image, Bands, scale: int = 10):
    '''Z-Score normalization'''
    for i, band in enumerate(Bands):
        band_img = Image.select(band)
        meanStd = get_meanStd(band_img, scale)
        if i == 0:
            result = band_img.subtract(meanStd.get('mean')).divide(meanStd.get('std'))
        else:
            result = result.addBands(band_img.subtract(meanStd.get('mean')).divide(meanStd.get('std')))
    return result

def get_histogram(Image: ee.Image, region, scale, histNum=1000):
    histogram = Image.reduceRegion(
        reducer=ee.Reducer.histogram(histNum),
        geometry=region,
        scale=scale,
        maxPixels=1e12,
        bestEffort=True)
    return histogram.get(Image.bandNames().get(0))

def HistBoundary(counts, y=100):
    '''Histogram boundary truncation'''
    boundary_value = int(sum(counts) / y)
    count_front = 0
    count_back = 0

    for index, count in enumerate(counts):
        if count_front + count >= boundary_value:
            index_front = index
            break
        else:
            count_front += count

    for count, index in zip(counts[::-1], range(len(counts))[::-1]):
        if count_back + count >= boundary_value:
            index_back = index
            break
        else:
            count_back += count

    return {'countFront': count_front, 'indexFront': index_front,
            'countBack': count_back, 'indexBack': index_back}


def GetHistAndBoundary(Image: ee.Image, region, scale, histNum=1000, y=100):
    '''Get histogram and boundaries for GEE'''
    histogram = get_histogram(Image, region, scale, histNum=histNum).getInfo()
    bin_centers, counts = np.array(histogram['bucketMeans']), np.array(histogram['histogram'])
    bounds = HistBoundary(counts, y=y)
    bin_centers = bin_centers[bounds['indexFront']:bounds['indexBack']]
    counts = counts[bounds['indexFront']:bounds['indexBack']]
    return bin_centers, counts, histogram['bucketWidth']

def histogramMatching(sourceImg, targetImg, AOI, source_bandsNames, target_bandsNames, Histscale=30, maxBuckets=256):
    '''Histogram matching'''
    def lookup(sourceHist, targetHist):
        source_values = sourceHist.slice(1, 0, 1).project([0])
        source_counts = sourceHist.slice(1, 1, 2).project([0]).divide(sourceHist.slice(1, 1, 2).get([-1]))
        target_values = targetHist.slice(1, 0, 1).project([0])
        target_counts = targetHist.slice(1, 1, 2).project([0]).divide(targetHist.slice(1, 1, 2).get([-1]))
        
        def _n(n):
            index = target_counts.gte(n).argmax()
            return target_values.get(index)
        
        y_values = source_counts.toList().map(_n)
        return {'x': source_values.toList(), 'y': y_values}

    assert len(source_bandsNames) == len(target_bandsNames), 'Band count mismatch between source and target images'

    args = {
        'reducer': ee.Reducer.autoHistogram(**{'maxBuckets': maxBuckets, 'cumulative': True}),
        'geometry': AOI,
        'scale': Histscale,
        'maxPixels': 1e13,
        'tileScale': 16
    }
    source = sourceImg.reduceRegion(**args)
    target = targetImg.updateMask(sourceImg.mask()).reduceRegion(**args)

    matched_bands = []
    for band_source, band_target in zip(source_bandsNames, target_bandsNames):
        lookup_table = lookup(source.getArray(band_source), target.getArray(band_target))
        matched_bands.append(sourceImg.select([band_source]).interpolate(**lookup_table))
    return ee.Image.cat(matched_bands)

def delBands(Image: ee.Image, *BandsNames):
    '''Remove bands from ee.Image'''
    bands = Image.bandNames()
    for band in BandsNames:
        bands = bands.remove(band)
    return Image.select(bands)

def replaceBands(Image1: ee.Image, Image2: ee.Image):
    '''Replace bands in one image with bands from another'''
    bands1 = Image1.bandNames()
    bands2 = Image2.bandNames()
    return Image1.select(bands1.removeAll(bands2)).addBands(Image2)

def clip_AOI(col, AOI):
    return col.clip(AOI)

def cut_geometryGEE(geometry, block_size: float = 0.05):
    '''Cut geometry into blocks'''
    bounds = ee.List(geometry.bounds().coordinates().get(0))
    width = ee.Number(ee.List(bounds.get(2)).get(0)).subtract(ee.Number(ee.List(bounds.get(0)).get(0)))
    height = ee.Number(ee.List(bounds.get(2)).get(1)).subtract(ee.Number(ee.List(bounds.get(0)).get(1)))
    num_rows = height.divide(block_size).ceil().getInfo()
    num_cols = width.divide(block_size).ceil().getInfo()

    def create_blocks(row, col):
        x_min = ee.Number(ee.List(bounds.get(0)).get(0)).add(col.multiply(block_size))
        y_min = ee.Number(ee.List(bounds.get(0)).get(1)).add(row.multiply(block_size))
        x_max = x_min.add(block_size)
        y_max = y_min.add(block_size)
        return ee.Geometry.Rectangle([x_min, y_min, x_max, y_max])

    block_list = []
    for row in trange(num_rows):
        for col in trange(num_cols):
            block = create_blocks(ee.Number(row), ee.Number(col))
            block_list.append(block)
    return block_list


def cut_geometry(geometry, block_size: float = 0.05):
    block_size = 0.05
    bounds = geometry.bounds().coordinates().getInfo()[0]

    width = bounds[2][0] - bounds[0][0]
    height = bounds[2][1] - bounds[0][1]
    num_rows = math.ceil(height / block_size)
    num_cols = math.ceil(width / block_size)

    def create_blocks(row, col):
        x_min = bounds[0][0] + col * block_size
        y_min = bounds[0][1] + row * block_size
        x_max = x_min + block_size
        y_max = y_min + block_size
        return ee.Geometry.Rectangle([x_min, y_min, x_max, y_max])

    # 生成方块列表
    block_list = []
    for row in trange(num_rows):
        for col in trange(num_cols):
            block = create_blocks(row, col)
            block_list.append(block)
    return block_list


def time_difference(col, middle_date):
    '''计算middle_date与col包含日期的差值'''
    time_difference = middle_date.difference(
        ee.Date(col.get('system:time_start')), 'days').abs()
    return col.set({'time_difference': time_difference})


def rm_nodata(col, AOI):
    allNone_num = col.select('VV').unmask(-99).eq(-99).reduceRegion(
        **{
            'geometry': AOI,
            'reducer': ee.Reducer.sum(),
            'scale': 10,
            'maxPixels': 1e12,
            'bestEffort': True,
        }).get('VV')
    return col.set({'numNodata': allNone_num})


def rename_band(img_path, new_names: list, rewrite=False):
    ds = gdal.Open(img_path)
    band_count = ds.RasterCount
    assert band_count == len(new_names), 'BnadNames length not match'
    for i in range(band_count):
        ds.GetRasterBand(i + 1).SetDescription(new_names[i])
    driver = gdal.GetDriverByName('GTiff')
    if rewrite:
        dst_ds = driver.CreateCopy(img_path, ds)
    else:
        DirName = os.path.dirname(img_path)
        BaseName = os.path.basename(img_path).split('.')[0] + '_Copy.' + os.path.basename(img_path).split('.')[1]
        dst_ds = driver.CreateCopy(os.path.join(DirName, BaseName), ds)
    dst_ds = None
    ds = None


def Geemap_export(fileDirname, collection=False, image=False, rename_image=True,
                  region=None, scale=10,
                  vector=False, keep_zip=True):


    if collection:
        geemap.ee_export_image_collection(collection,
                                          out_dir=os.path.dirname(fileDirname),
                                          format="ZIPPED_GEO_TIFF", region=region, scale=scale)
        print('collection save right')

    elif image:
        if os.path.exists(fileDirname):
            print('File already exists:{}'.format(fileDirname))
            pass
        else:
            geemap.ee_export_image(image,
                                   filename=fileDirname,
                                   scale=scale, region=region, file_per_band=False, timeout=3000)
            print('image save right')
            if rename_image:
                print('change image bandNames')
                rename_band(fileDirname, new_names=image.bandNames().getInfo(), rewrite=True)

    elif vector:
        if os.path.exists(fileDirname):
            print('File already exists:{}'.format(fileDirname))
            pass
        else:
            geemap.ee_export_vector(vector, fileDirname, selectors=None, verbose=True, keep_zip=keep_zip, timeout=300,
                                    proxies=None)
            print('vector save right')

    else:
        print('Erro:collection && image must have one False')


def load_image_collection(aoi, start_date, end_date, middle_date,
                          Filter=None, FilterSize=30):
    s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .filterBounds(aoi)
              .filterDate(start_date, end_date))
    s1_col_copy = copy.deepcopy(s1_col)

    s1_col = s1_col.map(partial(rm_nodata, AOI=aoi))
    if Filter:
        print('Begin Filter ...')
        if Filter == 'leesigma':
            s1_col = s1_col.map(leesigma(FilterSize))
        elif Filter == 'RefinedLee':
            s1_col = s1_col.map(RefinedLee)
        elif Filter == 'gammamap':
            s1_col = s1_col.map(gammamap(FilterSize))
        elif Filter == 'boxcar':
            s1_col = s1_col.map(boxcar(FilterSize))
        else:
            print('Wrong Filter')
    else:
        print('Without Filter')

    s1_col = s1_col.map(partial(time_difference, middle_date=middle_date))

    s1_descending = s1_col.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    s1_ascending = s1_col.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))

    filtered_collection_A = s1_ascending.filter(ee.Filter.eq('numNodata', 0))
    has_images_without_nodata_A = filtered_collection_A.size().eq(0)

    s1_ascending = ee.Algorithms.If(
        has_images_without_nodata_A,
        s1_ascending.median().reproject(s1_ascending.first().select(0).projection().crs(), None, 10).set(
            {'synthesis': 1}),
        filtered_collection_A.filter(ee.Filter.eq('time_difference',
                                                  filtered_collection_A.aggregate_min('time_difference'))).first().set(
            {'synthesis': 0})
    )

    filtered_collection_D = s1_descending.filter(ee.Filter.eq('numNodata', 0))
    has_images_without_nodata_D = filtered_collection_D.size().eq(0)
    s1_descending = ee.Algorithms.If(
        has_images_without_nodata_D,
        s1_descending.median().reproject(s1_descending.first().select(0).projection().crs(), None, 10).set(
            {'synthesis': 1}),
        filtered_collection_D.filter(ee.Filter.eq('time_difference',
                                                  filtered_collection_D.aggregate_min('time_difference'))).first().set(
            {'synthesis': 0})
    )

    return ee.Image(s1_ascending), ee.Image(s1_descending)  # ,s1_col_copy


# --获取子数据集(主要用于删除GEE Asset)
def get_asset_list(parent):
    parent_asset = ee.data.getAsset(parent)
    parent_id = parent_asset['name']
    parent_type = parent_asset['type']
    asset_list = []
    child_assets = ee.data.listAssets({'parent': parent_id})['assets']
    for child_asset in child_assets:
        child_id = child_asset['name']
        child_type = child_asset['type']
        if child_type in ['FOLDER', 'IMAGE_COLLECTION']:
            # Recursively call the function to get child assets
            asset_list.extend(get_asset_list(child_id))
        else:
            asset_list.append(child_id)
    return asset_list


# 删除数据集，包含子数据集（慎用）
def delete_asset_list(asset_list, save_parent=1):
    for asset in asset_list:
        ee.data.deleteAsset(asset)
    if save_parent:
        print('del,save parent')
    else:
        ee.data.deleteAsset(os.path.dirname(asset_list[0]))
        print('del parent')

try:
    from geeup import geeup
    def reload_variable(region,
                        scale=10,
                        save_dir='./test',
                        dest='users/xx/xx',
                        metaName='test.csv',
                        eeUser='@xx.com',
                        overwrite='yes',
                        delgeeUp=False,
                        **parms):
        if not os.path.isabs(save_dir):
            save_dir = os.path.join(os.getcwd(), os.path.basename(save_dir))
        save_dir = make_dir(save_dir)
        meta_csv = os.path.join(save_dir, metaName)
        for key, value in parms.items():
            fileName = os.path.join(save_dir, key + '.tif')
            Geemap_export(fileName, image=value, region=region, scale=scale)

        geeup.getmeta(save_dir, meta_csv)  # !geeup getmeta --input save_dir --metadata meta_csv
        geeup.upload(user=eeUser, source_path=save_dir, pyramiding="MEAN", mask=False, nodata_value=None,
                     metadata_path=meta_csv, destination_path=dest,
                     overwrite=overwrite)  # !geeup upload --source save_dir --dest dest -m meta_csv --nodata 0 -u eeUser
        Images = ee.ImageCollection(dest)
        for key in parms.keys():
            vars()[key] = Images.filter(ee.Filter.eq('id_no', ee.String(key))).first()
            parms[key] = vars()[key]
        if delgeeUp:
            asset_list = get_asset_list(dest)
            delete_asset_list(asset_list, save_parent=0)
        return parms
except:
    print('geeup not import')

def Select_imageNum(ImageCollection: ee.ImageCollection, i):
    return ee.Image(ImageCollection.toList(ImageCollection.size()).get(i))

def afn_normalize_by_maxes(img, scale=10):
    max_ = img.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=img.geometry(),
        scale=scale,
        maxPixels=1e12,
        bestEffort=True
    ).toImage().select(img.bandNames())

    min_ = img.reduceRegion(
        reducer=ee.Reducer.min(),
        geometry=img.geometry(),
        scale=scale,
        maxPixels=1e12,
        bestEffort=True
    ).toImage().select(img.bandNames())  #

    return img.subtract(min_).divide(max_.subtract(min_))


def my_slope_correction(s1_ascending, s1_descending,
                        AOI_buffer, DEM, model, Origin_scale,
                        DistorMethed='RS'):
    volumetric_dict = {}
    for image, orbitProperties_pass in zip([s1_ascending, s1_descending], ['ASCENDING', 'DESCENDING']):

        geom = image.geometry()
        proj = image.select(0).projection()

        azimuthEdge, rotationFromNorth, startpoint, endpoint, coordinates_dict = getASCCorners(image, AOI_buffer,
                                                                                               orbitProperties_pass)
        Heading = azimuthEdge.get('azimuth')
        # Heading_Rad = ee.Image.constant(Heading).multiply(np.pi / 180)

        s1_azimuth_across = ee.Number(Heading).subtract(90.0)
        theta_iRad = image.select('angle').multiply(np.pi / 180)  # 地面入射角度转为弧度
        phi_iRad = ee.Image.constant(s1_azimuth_across).multiply(np.pi / 180)  # 方位角转弧度

        def slop_aspect(elevation, proj, geom):
            alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(
                np.pi / 180).setDefaultProjection(proj).clip(geom)  # 坡度(与地面夹角)
            phi_sRad = ee.Terrain.aspect(elevation).select('aspect').multiply(
                np.pi / 180).setDefaultProjection(proj).clip(geom)  # 坡向角，(坡度陡峭度)坡与正北方向夹角(陡峭度)，从正北方向起算，顺时针计算角度
            phi_rRad = phi_iRad.subtract(phi_sRad)  # (飞行方向角度-坡度陡峭度)飞行方向与坡向之间的夹角

            alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()  # 距离向分解
            alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()  # 方位向分解
            return alpha_sRad, phi_sRad, alpha_rRad, alpha_azRad

        alpha_sRad, phi_sRad, alpha_rRad, alpha_azRad = slop_aspect(DEM, proj, geom)
        theta_liaRad = (alpha_azRad.cos().multiply(
            (theta_iRad.subtract(alpha_rRad)).cos())).acos()  # LIA
        theta_liaDeg = theta_liaRad.multiply(180 / np.pi)  # LIA转弧度

        sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))
        gamma0 = sigma0Pow.divide(theta_iRad.cos())  # 根据角度修订入射值
        gamma0dB = ee.Image.constant(10).multiply(gamma0.log10()).select(['VV', 'VH'],
                                                                         ['VV_gamma0', 'VH_gamma0'])  # 根据角度修订入射值
        ratio_gamma = (
            gamma0dB.select('VV_gamma0').subtract(gamma0dB.select('VH_gamma0')).rename('ratio_gamma0'))  # gamma极化相减

        def volumetric(model, theta_iRad, alpha_rRad, alpha_azRad):
            '''辐射斜率校正'''
            if model == 'volume':
                scf = volumetric_model_SCF(theta_iRad, alpha_rRad)
            if model == 'surface':
                scf = surface_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)

            gamma0_flat = gamma0.divide(scf)
            gamma0_flatDB = (ee.Image.constant(10).multiply(gamma0_flat.log10()).select(['VV', 'VH'], ['VV_gamma0flat',
                                                                                                       'VH_gamma0flat']))
            ratio_flat = (gamma0_flatDB.select('VV_gamma0flat').subtract(
                gamma0_flatDB.select('VH_gamma0flat')).rename('ratio_gamma0flat'))

            return {'scf': scf, 'gamma0_flat': gamma0_flat,
                    'gamma0_flatDB': gamma0_flatDB, 'ratio_flat': ratio_flat}

        if DistorMethed == 'RS':
            layover = alpha_rRad.gt(theta_iRad).rename('layover')
            ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)
            shadow = alpha_rRad.lt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad))).rename('shadow')
        elif DistorMethed == 'IJRS':
            layover = alpha_rRad.gt(theta_iRad).rename('layover')
            shadow = theta_liaDeg.gt(ee.Image.constant(85).multiply(np.pi / 180)).rename('shadow')
        elif DistorMethed == 'Wu':
            layover = theta_liaDeg.lt(ee.Image.constant(0).multiply(np.pi / 180)).rename('layover')
            shadow = theta_liaDeg.gt(ee.Image.constant(90).multiply(np.pi / 180)).rename('shadow')
        # RINDEX，
        else:
            raise Exception('DistorMethed is not right!')

        # combine layover and shadow
        no_data_maskrgb = rgbmask(image, layover=layover, shadow=shadow)
        slop_correction = volumetric(model, theta_iRad, alpha_rRad, alpha_azRad)

        image2 = (Eq_pixels(image.select('VV')).rename('VV_sigma0')
                  .addBands(Eq_pixels(image.select('VH')).rename('VH_sigma0'))
                  .addBands(Eq_pixels(image.select('angle')).rename('incAngle'))
                  .addBands(Eq_pixels(alpha_sRad.reproject(crs=proj, scale=Origin_scale)).rename('alpha_sRad'))
                  .addBands(Eq_pixels(alpha_rRad.reproject(crs=proj, scale=Origin_scale)).rename('alpha_rRad'))
                  .addBands(gamma0dB)
                  .addBands(Eq_pixels(slop_correction['gamma0_flat'].select('VV')).rename('VV_gamma0_flat'))
                  .addBands(Eq_pixels(slop_correction['gamma0_flat'].select('VH')).rename('VH_gamma0_flat'))
                  .addBands(
                            Eq_pixels(slop_correction['gamma0_flatDB'].select('VV_gamma0flat')).rename('VV_gamma0_flatDB'))
                  .addBands(
                            Eq_pixels(slop_correction['gamma0_flatDB'].select('VH_gamma0flat')).rename('VH_gamma0_flatDB'))
                  .addBands(Eq_pixels(layover).rename('layover'))
                  .addBands(Eq_pixels(shadow).rename('shadow'))
                  .addBands(no_data_maskrgb)
                  .addBands(Eq_pixels(DEM.setDefaultProjection(proj).clip(geom)).rename('height')))

        cal_image = (image2.addBands(ee.Image.pixelCoordinates(proj))
                     .addBands(ee.Image.pixelLonLat()).reproject(crs=proj, scale=Origin_scale)
                     .updateMask(image2.select('VV_sigma0').mask()).clip(AOI_buffer))

        Auxiliarylines = ee.Geometry.LineString([startpoint, endpoint])

        if orbitProperties_pass == 'ASCENDING':
            volumetric_dict['ASCENDING'] = cal_image
            volumetric_dict['ASCENDING_parms'] = {'s1_azimuth_across': s1_azimuth_across,
                                                  'coordinates_dict': coordinates_dict,
                                                  'Auxiliarylines': Auxiliarylines,
                                                  'orbitProperties_pass': orbitProperties_pass,
                                                  'proj': proj}
        elif orbitProperties_pass == 'DESCENDING':
            volumetric_dict['DESCENDING'] = cal_image
            volumetric_dict['DESCENDING_parms'] = {'s1_azimuth_across': s1_azimuth_across,
                                                   'coordinates_dict': coordinates_dict,
                                                   'Auxiliarylines': Auxiliarylines,
                                                   'orbitProperties_pass': orbitProperties_pass,
                                                   'proj': proj}

    return volumetric_dict


try:
    import rasterio as rio
    from rasterio.warp import calculate_default_transform
    def getTransform(tif_path, dst_crs="EPSG:4326"):
        with rio.open(tif_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
        return transform
except:
    print('rasterio not import')