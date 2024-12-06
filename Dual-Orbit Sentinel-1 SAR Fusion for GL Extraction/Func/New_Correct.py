import ee
import math
import numpy as np
from scipy.signal import argrelextrema
from tqdm import tqdm
import sys

EasyIndex = lambda Data, Index, *Keys: [Data[key][Index] for key in Keys]


def getASCCorners(image, AOI_buffer, orbitProperties_pass):
    coords = ee.Array(image.geometry().coordinates().get(0)).transpose()
    crdLons = ee.List(coords.toList().get(0))
    crdLats = ee.List(coords.toList().get(1))
    minLon = crdLons.sort().get(0)
    maxLon = crdLons.sort().get(-1)
    minLat = crdLats.sort().get(0)
    maxLat = crdLats.sort().get(-1)
    azimuth = (ee.Number(crdLons.get(crdLats.indexOf(minLat))).subtract(minLon).atan2(
        ee.Number(crdLats.get(crdLons.indexOf(minLon))).subtract(minLat))
               .multiply(180.0 / math.pi))

    if orbitProperties_pass == 'ASCENDING':
        azimuth = azimuth.add(270.0)
        rotationFromNorth = azimuth.subtract(360.0)
    elif orbitProperties_pass == 'DESCENDING':
        azimuth = azimuth.add(180.0)
        rotationFromNorth = azimuth.subtract(180.0)
    else:
        raise TypeError

    azimuthEdge = (ee.Feature(ee.Geometry.LineString([crdLons.get(crdLats.indexOf(minLat)), minLat, minLon,
                                                      crdLats.get(crdLons.indexOf(minLon))]),
                              {'azimuth': azimuth}).copyProperties(image))

    coords = ee.Array(image.clip(AOI_buffer).geometry().coordinates().get(0)).transpose()
    crdLons = ee.List(coords.toList().get(0))
    crdLats = ee.List(coords.toList().get(1))
    minLon = crdLons.sort().get(0)
    maxLon = crdLons.sort().get(-1)
    minLat = crdLats.sort().get(0)
    maxLat = crdLats.sort().get(-1)

    if orbitProperties_pass == 'ASCENDING':
        startpoint = ee.List([minLon, maxLat])
        endpoint = ee.List([maxLon, minLat])
    elif orbitProperties_pass == 'DESCENDING':
        startpoint = ee.List([maxLon, maxLat])
        endpoint = ee.List([minLon, minLat])

    coordinates_dict = {'crdLons': crdLons, 'crdLats': crdLats,
                        'minLon': minLon, 'maxLon': maxLon, 'minLat': minLat, 'maxLat': maxLat}

    return azimuthEdge, rotationFromNorth, startpoint, endpoint, coordinates_dict


def rgbmask(image, **parms):
    r = ee.Image.constant(0).select([0], ['red'])
    g = ee.Image.constant(0).select([0], ['green'])
    b = ee.Image.constant(0).select([0], ['blue'])

    for key, value in parms.items():
        lenName = value.bandNames().length().getInfo()
        if lenName:
            if key == 'layover' or key == 'Llayover':
                # print('r,key={}'.format(key))
                r = r.where(value, 255)
            if key == 'shadow':
                # print('g,key={}'.format(key))
                g = g.where(value, 255)
            if key == 'Rlayover':
                # print('b,key={}'.format(key))
                b = b.where(value, 255)
        else:
            continue
    return ee.Image.cat([r, g, b]).byte().updateMask(image.mask())


def angle2slope(angle):
    if type(angle) == ee.ee_number.Number:
        angle = angle.getInfo()
    if 0 < angle <= 90 or 180 < angle <= 270:
        if 180 < angle <= 270:
            angle = 90 - (angle - 180)
        arc = angle / 180 * math.pi
        slop = math.tan(arc)
    elif 90 < angle <= 180 or 270 < angle <= 360:
        if 90 < angle <= 180:
            angle = angle - 90
        elif 270 < angle <= 360:
            angle = angle - 270
        arc = angle / 180 * math.pi
        slop = -math.tan(arc)
    return slop


def AuxiliaryLine2Point(cal_image, s1_azimuth_across, coordinates_dict, Auxiliarylines, scale):
    K = angle2slope(s1_azimuth_across)
    Max_Lon = coordinates_dict['maxLon'].getInfo()
    Min_Lon = coordinates_dict['minLon'].getInfo()

    AuxiliaryPoints = reduce_tolist(cal_image.select(['longitude', 'latitude']).clip(Auxiliarylines),
                                    scale=scale).getInfo()
    Aux_lon = np.array(AuxiliaryPoints['longitude'])
    Aux_lon, indices_Aux_lon = np.unique(Aux_lon, return_index=True)
    Aux_lat = np.array(AuxiliaryPoints['latitude'])[indices_Aux_lon]

    Templist = []
    for X, Y in zip(Aux_lon, Aux_lat):
        C = Y - K * X
        Min_Lon_Y = K * Min_Lon + C
        Max_lon_Y = K * Max_Lon + C
        Templist.append(ee.Feature(ee.Geometry.LineString(
            [Min_Lon, Min_Lon_Y, Max_Lon, Max_lon_Y])))
    return Templist


def Eq_pixels(x): return ee.Image.constant(0).where(x, x).updateMask(x.mask())

def reduce_tolist(each, scale): return ee.Image(each).reduceRegion(
    reducer=ee.Reducer.toList(), geometry=each.geometry(), scale=scale, maxPixels=1e13)


def Line_Correct_old(cal_image, AOI, Templist, orbitProperties_pass, proj, scale: int, cal_image_scale: int):
    line_points_list = []
    LPassive_layover_linList = []
    RPassive_layover_linList = []
    Shadow_linList = []

    for eachLine in Templist:
        LineImg = cal_image.select(
            ['height', 'layover', 'shadow', 'incAngle', 'x', 'y', 'longitude', 'latitude']).clip(eachLine)
        LineImg_point = LineImg.reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=cal_image.geometry(),
            scale=scale,
            maxPixels=1e13)
        line_points_list.append(LineImg_point)
    list_of_dicts = ee.List(line_points_list).getInfo()  

    for PointDict in tqdm(list_of_dicts):
        if orbitProperties_pass == 'ASCENDING':
            order = np.argsort(PointDict['x'])
        elif orbitProperties_pass == 'DESCENDING':
            order = np.argsort(PointDict['x'])[::-1]
        PointDict = {k: np.array(v)[order] for k, v in PointDict.items()}
        PointDict['x'], PointDict['y'] = PointDict['x'] * \
                                         cal_image_scale, PointDict['y'] * cal_image_scale  # 像素行列10m分辨率，由proj得

        index_max = argrelextrema(PointDict['height'], np.greater)[0]
        Angle_max, Z_max, X_max, Y_max = EasyIndex(
            PointDict, index_max, 'incAngle', 'height', 'x', 'y')

        LPassive_layover = []
        RPassive_layover = []
        Passive_shadow = []
        for each in range(len(index_max)):

            rx = X_max[each]
            ry = Y_max[each]
            rh = Z_max[each]
            rindex = index_max[each]
            r_angle = Angle_max[each]

            if index_max[each] - 50 > 0:
                rangeIndex = range(index_max[each] - 50, index_max[each])
            else:
                rangeIndex = range(0, index_max[each])

            L_h, L_x, L_y, L_lon, L_lat, L_angle = EasyIndex(
                PointDict, rangeIndex, 'height', 'x', 'y', 'longitude', 'latitude', 'incAngle')

            Llay_angle_iRad = np.arctan(
                (rh - L_h) / np.sqrt(np.square(L_x - rx) + np.square(L_y - ry)))
            Llay_angle = Llay_angle_iRad * 180 / math.pi
            index_Llay = np.where(Llay_angle > r_angle)[0]

            if len(index_Llay) != 0:
                tlon_Llay = L_lon[index_Llay]
                tlat_Llay = L_lat[index_Llay]
                LlayFeatureCollection = ee.FeatureCollection([ee.Feature(
                    ee.Geometry.Point(x, y), {'values': 1}) for x, y in zip(tlon_Llay, tlat_Llay)])
                LPassive_layover.append(
                    LlayFeatureCollection.reduceToImage(['values'], 'mean'))

            if index_max[each] + 50 < len(PointDict['x']):
                rangeIndex = range(index_max[each] + 1, index_max[each] + 50)
            else:
                rangeIndex = range(index_max[each] + 1, len(PointDict['x']))

            R_h, R_x, R_y, R_lon, R_lat = EasyIndex(
                PointDict, rangeIndex, 'height', 'x', 'y', 'longitude', 'latitude')
            R_angle_iRad = np.arctan(
                (rh - R_h) / np.sqrt(np.square(R_x - rx) + np.square(R_y - ry)) + sys.float_info.min)
            R_angle = R_angle_iRad * 180 / math.pi
            index_Shadow = np.where(R_angle > (90 - r_angle))[0]

            if len(index_Shadow) != 0:
                tlon_Shadow = R_lon[index_Shadow]
                tlat_Shadow = R_lat[index_Shadow]
                ShadowFeatureCollection = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(
                    x, y), {'values': 1}) for x, y in zip(tlon_Shadow, tlat_Shadow)])
                Passive_shadow.append(
                    ShadowFeatureCollection.reduceToImage(['values'], 'mean'))

            if len(index_Llay) != 0:
                layoverM_x, layoverM_y, layoverM_h, layoverM_angle = \
                    L_x[index_Llay[-1]], L_y[index_Llay[-1]], L_h[index_Llay[-1]], L_angle[index_Llay[-1]]  # 起算点
                Rlay_angle_iRad = np.arctan(
                    (R_h - layoverM_h) / np.sqrt(np.square(R_x - layoverM_x) + np.square(R_y - layoverM_y)))
                Rlay_angle = Rlay_angle_iRad * 180 / math.pi
                index_Rlayover = np.where(Rlay_angle > layoverM_angle)[0]
                if len(index_Rlayover) != 0:
                    tlon_RLay = R_lon[index_Rlayover]
                    tlat_RLay = R_lat[index_Rlayover]
                    RLayFeatureCollection = ee.FeatureCollection([ee.Feature(
                        ee.Geometry.Point(x, y), {'values': 1}) for x, y in zip(tlon_RLay, tlat_RLay)])
                    RPassive_layover.append(
                        RLayFeatureCollection.reduceToImage(['values'], 'mean'))

        if len(LPassive_layover) != 0:
            aggregated_image = ee.ImageCollection(
                LPassive_layover).mosaic().reproject(crs=proj, scale=scale)
            LPassive_layover_linList.append(aggregated_image)

        if len(RPassive_layover) != 0:
            aggregated_image = ee.ImageCollection(
                RPassive_layover).mosaic().reproject(crs=proj, scale=scale)
            RPassive_layover_linList.append(aggregated_image)

        if len(Passive_shadow) != 0:
            aggregated_image = ee.ImageCollection(
                Passive_shadow).mosaic().reproject(crs=proj, scale=scale)
            Shadow_linList.append(aggregated_image)

    LeftLayover = ee.ImageCollection(LPassive_layover_linList).mosaic().reproject(crs=proj, scale=scale).clip(AOI)
    RightLayover = ee.ImageCollection(RPassive_layover_linList).mosaic().reproject(crs=proj, scale=scale).clip(AOI)
    Shadow = ee.ImageCollection(Shadow_linList).mosaic().reproject(crs=proj, scale=scale).clip(AOI)

    return LeftLayover.toInt8(), RightLayover.toInt8(), Shadow.toInt8()


def Line_Correct(cal_image, AOI, Templist, orbitProperties_pass, proj, scale: int, cal_image_scale: int,
                 filt_distance=False, save_peak=False, line_points_connect=False, Peak_Llay=False, Peak_shdow=False,
                 Peak_Rlay=False):

    line_points_list = []
    LPassive_layover_linList = []
    RPassive_layover_linList = []
    Shadow_linList = []
    PeakPoint_list = []

    for eachLine in tqdm(Templist):
        LineImg = cal_image.select(
            ['height', 'layover', 'shadow', 'incAngle', 'alpha_sRad', 'x', 'y', 'longitude', 'latitude']).clip(eachLine)
        ptsDict = LineImg.reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=cal_image.geometry(),
            scale=scale,
            maxPixels=1e13)
        if filt_distance:
            lons = ee.List(ptsDict.get('longitude'))
            lats = ee.List(ptsDict.get('latitude'))
            Point_list = ee.FeatureCollection(lons.zip(lats).map(lambda xy: ee.Feature(ee.Geometry.Point(xy))))

            Point_list = Point_list.map(lambda f: f.set('dis', eachLine.geometry().distance(f.geometry())))
            distances = ee.List(Point_list.reduceColumns(ee.Reducer.toList(), ['dis']).get('list'))

            filteredDistances = ee.List(distances.filter(ee.Filter.lt('item', 30)))

            Index_filter = filteredDistances.map(lambda x: distances.indexOf(x))
            ptsDict = ptsDict.map(lambda k, v: Index_filter.map(lambda x: ee.List(v).get(x))).set('Distance',
                                                                                                  filteredDistances)
        line_points_list.append(ptsDict)

    list_of_dicts = ee.List(line_points_list).getInfo()  
    print('line_points_list={}'.format(sum([len(list_of_dicts[i]['longitude']) for i in range(len(list_of_dicts))])))

    for PointDict in tqdm(list_of_dicts):
        if orbitProperties_pass == 'ASCENDING':
            order = np.argsort(PointDict['longitude'])
        elif orbitProperties_pass == 'DESCENDING':
            order = np.argsort(PointDict['longitude'])[::-1]

        PointDict = {k: np.array(v)[order] for k, v in PointDict.items()}
        PointDict['x'], PointDict['y'] = PointDict['x'] * cal_image_scale, PointDict[
            'y'] * cal_image_scale  

        # 寻找入射线上DEM的极大值点
        index_max = argrelextrema(PointDict['height'], np.greater)[0]
        if len(index_max) != 0:
            Angle_max, Z_max, X_max, Y_max, Lon_max, Lat_max = EasyIndex(
                PointDict, index_max, 'incAngle', 'height', 'x', 'y', 'longitude', 'latitude')

            if save_peak:
                for x, y in zip(Lon_max, Lat_max):
                    print('len={}'.format(len(Lon_max)))
                    PeakPoint_list.append(ee.Feature(ee.Geometry.Point(x, y), {'values': 1}))
  
            LPassive_layover = []
            RPassive_layover = []
            Passive_shadow = []

            for each in range(len(index_max)):

                rx = X_max[each]
                ry = Y_max[each]
                rh = Z_max[each]
                rlon = Lon_max[each]
                rlat = Lat_max[each]
                rindex = index_max[each]
                r_angle = Angle_max[each]

                Pixels_cal = 900 // scale

                if index_max[each] - Pixels_cal > 0:
                    rangeIndex = range(rindex - Pixels_cal, rindex)
                else:
                    rangeIndex = range(0, rindex)
                
                PointDict['Grad_alpha_sRad'] = np.insert(np.diff(PointDict['alpha_sRad']), 0, 0)
                L_h, L_x, L_y, L_lon, L_lat, L_angle, L_Grad_alpha_sRad = EasyIndex(
                    PointDict, rangeIndex, 'height', 'x', 'y', 'longitude', 'latitude', 'incAngle', 'Grad_alpha_sRad')

                Llay_angle_iRad = np.arctan((rh - L_h) / np.sqrt(np.square(L_x - rx) + np.square(L_y - ry)))
                Llay_angle = Llay_angle_iRad * 180 / math.pi

                index_Llay = np.where(Llay_angle > r_angle)[0]

                if len(index_Llay) != 0:
                    if line_points_connect:
                        index_Llay = range(np.where(Llay_angle > r_angle)[0][0], len(Llay_angle))
                    tlon_Llay = L_lon[index_Llay]
                    tlat_Llay = L_lat[index_Llay]
                    if Peak_Llay:
                        tlon_Llay = np.append(tlon_Llay, rlon)
                        tlat_Llay = np.append(tlat_Llay, rlat)

                    LlayFeatureCollection = ee.FeatureCollection([ee.Feature(
                        ee.Geometry.Point(x, y), {'values': 1}) for x, y in zip(tlon_Llay, tlat_Llay)])

                    LPassive_layover.append(
                        LlayFeatureCollection.reduceToImage(['values'], 'mean'))

                if index_max[each] + Pixels_cal < len(PointDict['x']):
                    rangeIndex = range(index_max[each] + 1, index_max[each] + Pixels_cal)
                else:
                    rangeIndex = range(index_max[each] + 1, len(PointDict['x']))

                R_h, R_x, R_y, R_lon, R_lat = EasyIndex(
                    PointDict, rangeIndex, 'height', 'x', 'y', 'longitude', 'latitude')
                R_angle_iRad = np.arctan(
                    (rh - R_h) / np.sqrt(np.square(R_x - rx) + np.square(R_y - ry)) + sys.float_info.min)
                R_angle = R_angle_iRad * 180 / math.pi
                index_Shadow = np.where(R_angle > (90 - r_angle))[0]

                if len(index_Shadow) != 0:
                    if line_points_connect:
                        index_Shadow = range(0, np.where(R_angle > (90 - r_angle))[0][-1])
                    tlon_Shadow = R_lon[index_Shadow]
                    tlat_Shadow = R_lat[index_Shadow]
                    if Peak_shdow:
                        tlon_Shadow = np.append(tlon_Shadow, rlon)
                        tlat_Shadow = np.append(tlat_Shadow, rlat)
                    ShadowFeatureCollection = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(
                        x, y), {'values': 1}) for x, y in zip(tlon_Shadow, tlat_Shadow)])
                    Passive_shadow.append(
                        ShadowFeatureCollection.reduceToImage(['values'], 'mean'))

                if len(index_Llay) != 0:
                    try:
                        L_bottom = index_Llay[np.argmax(L_Grad_alpha_sRad[index_Llay])]
                        layoverM_x, layoverM_y = rx, ry
                        layoverM_h = L_h[L_bottom]  
                        layoverM_angle = r_angle
                    except:
                        raise IndexError('each={}'.format(each))

                    Rlay_angle_iRad = np.arctan(
                        (R_h - layoverM_h) / np.sqrt(np.square(R_x - layoverM_x) + np.square(R_y - layoverM_y)))
                    Rlay_angle = Rlay_angle_iRad * 180 / math.pi
                    index_Rlayover = np.where(Rlay_angle > layoverM_angle)[0]

                    if len(index_Rlayover) != 0:
                        if line_points_connect:
                            index_Rlayover = range(0, np.where(Rlay_angle > layoverM_angle)[0][-1])

                        tlon_RLay = R_lon[index_Rlayover]
                        tlat_RLay = R_lat[index_Rlayover]
                        if Peak_Rlay:
                            tlon_RLay = np.append(tlon_RLay, rlon)
                            tlat_RLay = np.append(tlat_RLay, rlat)

                        RLayFeatureCollection = ee.FeatureCollection([ee.Feature(
                            ee.Geometry.Point(x, y), {'values': 1}) for x, y in zip(tlon_RLay, tlat_RLay)])
                        RPassive_layover.append(
                            RLayFeatureCollection.reduceToImage(['values'], 'mean'))

                if len(LPassive_layover) != 0:
                    aggregated_image = ee.ImageCollection(
                        LPassive_layover).mosaic().reproject(crs=proj, scale=scale)
                    LPassive_layover_linList.append(aggregated_image)

                if len(RPassive_layover) != 0:
                    aggregated_image = ee.ImageCollection(
                        RPassive_layover).mosaic().reproject(crs=proj, scale=scale)
                    RPassive_layover_linList.append(aggregated_image)

                if len(Passive_shadow) != 0:
                    aggregated_image = ee.ImageCollection(
                        Passive_shadow).mosaic().reproject(crs=proj, scale=scale)
                    Shadow_linList.append(aggregated_image)

    LeftLayover = ee.ImageCollection(LPassive_layover_linList).mosaic().reproject(crs=proj, scale=scale).clip(AOI)
    RightLayover = ee.ImageCollection(RPassive_layover_linList).mosaic().reproject(crs=proj, scale=scale).clip(AOI)
    Shadow = ee.ImageCollection(Shadow_linList).mosaic().reproject(crs=proj, scale=scale).clip(AOI)
    if save_peak:
        return LeftLayover.toInt8(), RightLayover.toInt8(), Shadow.toInt8(), PeakPoint_list
    else:
        return LeftLayover.toInt8(), RightLayover.toInt8(), Shadow.toInt8()

def Line_Correct2(cal_image, AOI_buffer, Templist, orbitProperties_pass, proj, scale: int, cal_image_scale: int):
    line_points_list = []
    for eachLine in tqdm(Templist):
        LineImg = cal_image.select(
            ['height', 'layover', 'shadow', 'incAngle', 'x', 'y', 'longitude', 'latitude']).clip(eachLine)
        LineImg_point = LineImg.reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=cal_image.geometry(),
            scale=scale,
            maxPixels=1e13)
        line_points_list.append(LineImg_point)
    list_of_dicts = ee.List(line_points_list).getInfo()  
    r_lon_sum = []
    r_lat_sum = []
    l_lon_sum = []
    l_lat_sum = []
    sh_lon_sum = []
    sh_lat_sum = []
    for PointDict in tqdm(list_of_dicts):
        if orbitProperties_pass == 'ASCENDING':
            order = np.argsort(PointDict['x'])
        elif orbitProperties_pass == 'DESCENDING':
            order = np.argsort(PointDict['x'])[::-1]
        PointDict = {k: np.array(v)[order] for k, v in PointDict.items()}  # 将视线上的像元按照x，从小到大排序
        PointDict['x'], PointDict['y'] = PointDict['x'] * \
                                         cal_image_scale, PointDict['y'] * cal_image_scale  # 像素行列10m分辨率，由proj得
        EasyIndex = lambda Data, Index, *Keys: [Data[key][Index] for key in Keys]
        if len(np.where(PointDict['shadow'] == 1)[0]) != 0:
            shadow_grad_fn = lambda shadow_array: [shadow_array[i + 1] - shadow_array[i] for i in
                                                   range(len(shadow_array) - 1)]
            PointDict_shadow_padding = np.insert(PointDict['shadow'], 0, 0)
            shadow_grad = np.array(shadow_grad_fn(PointDict_shadow_padding))
            if PointDict_shadow_padding[-1] == 1:
                shadow_start_index = np.where(shadow_grad == 1)[0][:-1]
            else:
                shadow_start_index = np.where(shadow_grad == 1)[0]
            shadow_range_index = np.where(shadow_grad == -1)[0]
            sh_h, sh_x, s_y, sh_lon, sh_lat, sh_angle = EasyIndex(
                PointDict, shadow_start_index, 'height', 'x', 'y', 'longitude', 'latitude', 'incAngle')

            if len(shadow_start_index) != 0:
                for i in range(shadow_start_index.size):
                    start_sh_index = shadow_start_index[i]
                    range_sh_index = shadow_range_index[i]
                    index_Shadow = []
                    if range_sh_index + 50 < len(PointDict['shadow']):
                        rangeIndex_shadow = range(range_sh_index, range_sh_index + 50)
                    else:
                        rangeIndex_shadow = range(range_sh_index, len(PointDict['shadow']))
                    if len(rangeIndex_shadow) != 0:
                        sh_Range_h, sh_Range_x, sh_Range_y, sh_Range_lon, sh_Range_lat = EasyIndex(
                            PointDict, rangeIndex_shadow, 'height', 'x', 'y', 'longitude', 'latitude')
                        shadow_angle_iRad = np.arctan((sh_h[i] - sh_Range_h) / np.sqrt(
                            np.square(sh_Range_x - sh_x[i]) + np.square(sh_Range_y - s_y[i])))
                        shadow_angle = shadow_angle_iRad * 180 / math.pi
                        index_Shadow = np.where(shadow_angle > (90 - sh_angle[i]))[0]

                    if len(index_Shadow) != 0:
                        tlon_shadow = sh_Range_lon[index_Shadow]
                        tlat_shadow = sh_Range_lat[index_Shadow]
                        for j in range(len(tlon_shadow)):
                            sh_lon_sum.append(tlon_shadow[j])
                            sh_lat_sum.append(tlat_shadow[j])

        if len(np.where(PointDict['layover'] == 1)[0]) != 0:
            layover_grad_fn = lambda lay_array: [lay_array[i + 1] - lay_array[i] for i in range(len(lay_array) - 1)]
            PointDict_lay_padding = np.insert(PointDict['layover'], 0, 0)
            layover_grad = np.array(layover_grad_fn(PointDict_lay_padding))
            if PointDict_lay_padding[-1] == 1:
                rlayover_start_index = np.where(layover_grad == 1)[0][:-1]
            else:
                rlayover_start_index = np.where(layover_grad == 1)[0]
            rlayover_range_index = np.where(layover_grad == -1)[0]
            s_h, s_x, s_y, s_lon, s_lat, s_angle = EasyIndex(
                PointDict, rlayover_start_index, 'height', 'x', 'y', 'longitude', 'latitude', 'incAngle')
            d_h, d_x, d_y, d_lon, d_lat, d_angle = EasyIndex(
                PointDict, rlayover_range_index, 'height', 'x', 'y', 'longitude', 'latitude', 'incAngle')
            for i in range(rlayover_start_index.size):
                start_index = rlayover_start_index[i]
                range_index = rlayover_range_index[i]
                index_Rlayover = []
                index_Llayover = []
                if range_index + 50 < len(PointDict['layover']):
                    rangeIndex = range(range_index, range_index + 50)
                else:
                    rangeIndex = range(range_index, len(PointDict['layover']))

                if start_index - 50 > 0:
                    rangeIndex_l = range(start_index - 50, start_index)
                else:
                    rangeIndex_l = range(start_index)
                if len(rangeIndex) != 0:
                    r_Range_h, r_Range_x, r_Range_y, r_Range_lon, r_Range_lat = EasyIndex(
                        PointDict, rangeIndex, 'height', 'x', 'y', 'longitude', 'latitude')
                    Rlay_start_iRad = np.arctan(
                        (r_Range_h - s_h[i]) / np.sqrt(np.square(r_Range_x - s_x[i]) + np.square(r_Range_y - s_y[i])))
                    Rlay_end_iRad = np.arctan(
                        (r_Range_h - d_h[i]) / np.sqrt(np.square(r_Range_x - d_x[i]) + np.square(r_Range_y - d_y[i])))
                    Rlay_start = Rlay_start_iRad * 180 / math.pi
                    Rlay_end = Rlay_end_iRad * 180 / math.pi
                    index_Rlayover = np.where(np.logical_and(Rlay_start > s_angle[i], Rlay_end < d_angle[i]))[0]

                if len(rangeIndex_l) != 0:
                    l_Range_h, l_Range_x, l_Range_y, l_Range_lon, l_Range_lat = EasyIndex(
                        PointDict, rangeIndex_l, 'height', 'x', 'y', 'longitude', 'latitude')
                    Llay_start_iRad = np.arctan(
                        (s_h[i] - l_Range_h) / np.sqrt(np.square(s_x[i] - l_Range_x) + np.square(s_y[i] - l_Range_y)))
                    Llay_end_iRad = np.arctan(
                        (d_h[i] - l_Range_h) / np.sqrt(np.square(d_x[i] - l_Range_x) + np.square(d_y[i] - l_Range_y)))
                    Llay_start = Llay_start_iRad * 180 / math.pi
                    Llay_end = Llay_end_iRad * 180 / math.pi
                    index_Llayover = np.where(np.logical_and(Llay_start < s_angle[i], Llay_end > d_angle[i]))[0]

                if len(index_Rlayover) != 0:
                    tlon_RLay = r_Range_lon[index_Rlayover]
                    tlat_RLay = r_Range_lat[index_Rlayover]
                    for j in range(len(tlat_RLay)):
                        r_lon_sum.append(tlon_RLay[j])
                        r_lat_sum.append(tlat_RLay[j])
                if len(index_Llayover) != 0:
                    tlon_LLay = l_Range_lon[index_Llayover]
                    tlat_LLay = l_Range_lat[index_Llayover]
                    for j in range(len(tlat_LLay)):
                        l_lon_sum.append(tlon_LLay[j])
                        l_lat_sum.append(tlat_LLay[j])

    rlay_featurecollection = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point(x, y), {'values': 5}) for x, y in zip(r_lon_sum, r_lat_sum)])
    shadow_featurecollection = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point(x, y), {'values': 3}) for x, y in zip(sh_lon_sum, sh_lat_sum)])
    llay_featurecollection = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point(x, y), {'values': 4}) for x, y in zip(l_lon_sum, l_lat_sum)])
    image_rlayover = ee.Image().paint(rlay_featurecollection, 'values')
    image_llayover = ee.Image().paint(llay_featurecollection, 'values')
    image_shadow = ee.Image().paint(shadow_featurecollection, 'values')
    img_rlayover = image_rlayover.clip(AOI_buffer).reproject(crs=proj, scale=scale)
    img_llayover = image_llayover.clip(AOI_buffer).reproject(crs=proj, scale=scale)
    img_shadow = image_shadow.clip(AOI_buffer).reproject(crs=proj, scale=scale)
    # passive_img = ee.ImageCollection([img_rlayover, img_llayover, img_shadow]).mosaic()
    return img_rlayover.toInt8(), img_llayover.toInt8(), img_shadow.toInt8()

