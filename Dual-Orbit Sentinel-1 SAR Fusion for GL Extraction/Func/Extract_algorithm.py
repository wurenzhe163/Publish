import ee
from functools import partial
from skimage.filters import threshold_minimum
from scipy import ndimage as ndi
import numpy as np
import geemap
import geopandas as gpd
import pandas as pd
from .Basic_tools import Open_close,calculate_iou
import os,sys

class img_sharp(object):
    @staticmethod
    def DoG(Img:ee.Image,fat_radius:int=3,fat_sigma:float=1.,
            skinny_radius:int=3,skinny_sigma:float=0.5):
        '''Difference of Gaussians (DoG)'''
        # Create the Difference of Gaussians (DoG) kernel
        fat = ee.Kernel.gaussian(radius=fat_radius, sigma=fat_sigma, units='pixels')
        skinny = ee.Kernel.gaussian(radius=skinny_radius, sigma=skinny_sigma, units='pixels')

        # Convolve the image with the Gaussian kernels
        convolved_fat = Img.convolve(fat)
        convolved_skinny = Img.convolve(skinny)

        return convolved_fat.subtract(convolved_skinny)

    @staticmethod
    def Laplacian(Img:ee.Image,normalize:bool=True):
        '''Laplacian'''
        # Create the Laplacian kernel
        kernel = ee.Kernel.laplacian8(normalize=normalize)

        # Convolve the image with the Laplacian kernel
        return Img.convolve(kernel)

    @staticmethod
    def Laplacian_of_Gaussian(Img:ee.Image,radius:int=3,sigma:float=1.):
        '''Laplacian of Gaussian'''

class Polarization_comb(object):
    pass

class Cluster_extract(object):
    @staticmethod
    def afn_Kmeans(inputImg, defaultStudyArea, numberOfUnsupervisedClusters=2, nativeScaleOfImage=10, numPixels=1000):
        training = inputImg.sample(region=defaultStudyArea, scale=nativeScaleOfImage, numPixels=numPixels)
        cluster = ee.Clusterer.wekaKMeans(numberOfUnsupervisedClusters).train(training)
        toexport = inputImg.cluster(cluster)
        clusterUnsup = toexport.select(0).rename('unsupervisedClass')
        return clusterUnsup

    @staticmethod
    def afn_Cobweb(inputImg, defaultStudyArea, cutoff=0.004, nativeScaleOfImage=10, numPixels=1000):
        training = inputImg.sample(region=defaultStudyArea, scale=nativeScaleOfImage, numPixels=numPixels)
        cluster = ee.Clusterer.wekaCobweb(cutoff=cutoff).train(training)
        toexport = inputImg.cluster(cluster)
        clusterUnsup = toexport.select(0).rename('unsupervisedClass')
        return clusterUnsup

    @staticmethod
    def afn_Xmeans(inputImg, defaultStudyArea, numberOfUnsupervisedClusters=2, nativeScaleOfImage=10, numPixels=1000):
        training = inputImg.sample(region=defaultStudyArea, scale=nativeScaleOfImage, numPixels=numPixels)
        cluster = ee.Clusterer.wekaXMeans(maxClusters=numberOfUnsupervisedClusters).train(training)
        toexport = inputImg.cluster(cluster)
        clusterUnsup = toexport.select(0).rename('unsupervisedClass')
        return clusterUnsup

    @staticmethod
    def afn_LVQ(inputImg, defaultStudyArea, numberOfUnsupervisedClusters=2, nativeScaleOfImage=10, numPixels=1000):
        training = inputImg.sample(region=defaultStudyArea, scale=nativeScaleOfImage, numPixels=numPixels)
        cluster = ee.Clusterer.wekaLVQ(numClusters=numberOfUnsupervisedClusters).train(training)
        toexport = inputImg.cluster(cluster)
        clusterUnsup = toexport.select(0).rename('unsupervisedClass')
        return clusterUnsup

    @staticmethod
    def afn_CascadeKMeans(inputImg, defaultStudyArea, numberOfUnsupervisedClusters=2, nativeScaleOfImage=10,
                          numPixels=1000):
        training = inputImg.sample(region=defaultStudyArea, scale=nativeScaleOfImage, numPixels=numPixels)
        cluster = ee.Clusterer.wekaCascadeKMeans(maxClusters=numberOfUnsupervisedClusters).train(training)
        toexport = inputImg.cluster(cluster)
        clusterUnsup = toexport.select(0).rename('unsupervisedClass')
        return clusterUnsup

    @staticmethod
    def afn_SNIC(imageOriginal, SuperPixelSize=10, Compactness=1, Connectivity=4, NeighborhoodSize=20,
                 SeedShape='square'):
        theSeeds = ee.Algorithms.Image.Segmentation.seedGrid(SuperPixelSize, SeedShape)

        snic2 = ee.Algorithms.Image.Segmentation.SNIC(image=imageOriginal,
                                                      size=SuperPixelSize,
                                                      compactness=Compactness,
                                                      connectivity=Connectivity,
                                                      neighborhoodSize=NeighborhoodSize,
                                                      seeds=theSeeds)
        theStack = snic2.addBands(theSeeds)
        return theStack

class Adaptive_threshold(object):
    @staticmethod
    def afn_otsu(histogram):
        counts = ee.Array(ee.Dictionary(histogram).get('histogram'))
        means = ee.Array(ee.Dictionary(histogram).get('bucketMeans'))
        size = means.length().get([0])
        total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
        sum_ = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
        mean_ = sum_.divide(total)
        indices = ee.List.sequence(1, size)

        def calc_bss(i, sum_, mean_):
            aCounts = counts.slice(0, ee.Number(i))
            aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0])
            aMeans = means.slice(0, ee.Number(i))
            aMean = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount)
            bCount = total.subtract(aCount)
            bMean = sum_.subtract(aCount.multiply(aMean)).divide(bCount)
            return aCount.multiply(aMean.subtract(mean_).pow(2)).add(bCount.multiply(bMean.subtract(mean_).pow(2)))

        bss = indices.map(partial(calc_bss, sum_=sum_, mean_=mean_))

        return means.sort(bss).get([-1])

    @staticmethod
    def afn_histPeak(img, region=None,default_value=1):
        img_numpy = geemap.ee_to_numpy(img, region=region,default_value=default_value)  # region必须是矩形
        threshold = threshold_minimum(img_numpy)
        return threshold

    @staticmethod
    def my_threshold_minimum(bin_centers, counts,max_num_iter = 10000):
        def find_local_maxima_idx(hist):
            maximum_idxs = list()
            direction = 1
            for i in range(hist.shape[0] - 1):
                if direction > 0:
                    if hist[i + 1] < hist[i]:
                        direction = -1
                        maximum_idxs.append(i)
                else:
                    if hist[i + 1] > hist[i]:
                        direction = 1
            return maximum_idxs

        smooth_hist = counts.astype('float32', copy=False)
        for counter in range(max_num_iter):
            smooth_hist = ndi.uniform_filter1d(smooth_hist, 3)
            maximum_idxs = find_local_maxima_idx(smooth_hist)
            if len(maximum_idxs) < 3:
                break

        if len(maximum_idxs) != 2:
            raise RuntimeError('Unable to find two maxima in histogram')
        elif counter == max_num_iter - 1:
            raise RuntimeError('Maximum iteration reached for histogram' 'smoothing')

        # Find lowest point between the maxima
        threshold_idx = np.argmin(smooth_hist[maximum_idxs[0]:maximum_idxs[1] + 1])
        return bin_centers[maximum_idxs[0] + threshold_idx]

    @staticmethod
    def my_threshold_yen(bin_centers, counts):
        # Calculate probability mass function
        pmf = counts.astype('float32', copy=False) / counts.sum()
        P1 = np.cumsum(pmf)  # Cumulative normalized histogram
        P1_sq = np.cumsum(pmf ** 2)
        # Get cumsum calculated from end of squared array:
        P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
        # P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid
        # '-inf' in crit. ImageJ Yen implementation replaces those values by zero.
        crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) *
                      (P1[:-1] * (1.0 - P1[:-1])) ** 2)
        return bin_centers[crit.argmax()]

    @staticmethod
    def my_threshold_isodata(bin_centers, counts, bin_width , returnAll=False):
        '''
        https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_isodata
        '''
        counts = counts.astype('float32', copy=False)
        csuml = np.cumsum(counts)
        csumh = csuml[-1] - csuml

        # intensity_sum contains the total pixel intensity from each bin
        intensity_sum = counts * bin_centers
        csum_intensity = np.cumsum(intensity_sum)
        lower = csum_intensity[:-1] / csuml[:-1]
        higher = (csum_intensity[-1] - csum_intensity[:-1]) / csumh[:-1]
        higher = (csum_intensity[-1] - csum_intensity[:-1]) / (csumh[:-1]+sys.float_info.min)
        all_mean = (lower + higher) / 2.0
        distances = all_mean - bin_centers[:-1]
        thresholds = bin_centers[:-1][(distances >= 0) & (distances < bin_width)]
        if len(thresholds) == 0:
            thresholds = bin_centers[:-1][distances >= 0][-1]
            thresholds = [bin_centers[:-1][distances >= 0][-1]]
        if returnAll:
            return thresholds
        else:
            return thresholds[0]

class Supervis_classify(object):
    pass

class Reprocess(object):
    '''图像后处理算法'''
    @staticmethod
    def Open_close(img, radius=10):
        uniformKernel = ee.Kernel.square(**{'radius': radius, 'units': 'meters'})
        min = img.reduceNeighborhood(**{'reducer': ee.Reducer.min(), 'kernel': uniformKernel})
        Openning = min.reduceNeighborhood(**{'reducer': ee.Reducer.max(), 'kernel': uniformKernel})
        max = Openning.reduceNeighborhood(**{'reducer': ee.Reducer.max(), 'kernel': uniformKernel})
        Closing = max.reduceNeighborhood(**{'reducer': ee.Reducer.min(), 'kernel': uniformKernel})
        return Closing

    @staticmethod
    def image2vector(result, resultband=0, radius=10,GLarea=1., scale=10,FilterBound=None, del_maxcount=False):

        Closing_result = Reprocess.Open_close(result.select(resultband), radius = radius)
        if GLarea > 20:
            Vectors = Closing_result.select(0).reduceToVectors(scale=scale*3, geometryType='polygon',
                                                               eightConnected=True)
        else:
            Vectors = Closing_result.select(0).reduceToVectors(scale=scale, geometryType='polygon',
                                                               eightConnected=True)
        if del_maxcount:
            Max_count = Vectors.aggregate_max('count')
            Vectors = Vectors.filterMetadata('count', 'not_equals', Max_count)
        Extract = Vectors.filterBounds(FilterBound)
        Union_ex = ee.Feature(Extract.union(1).first())

        return Union_ex

class save_parms(object):
    @staticmethod
    def save_log(log, mode='gpd', crs='EPSG:4326', logname='log.csv', shapname='log.shp'):
        if os.path.exists(logname):
            log.to_csv(logname, mode='a', index=False, header=0)
        else:
            log.to_csv(logname, mode='w', index=False)
            log.drop('geometry',axis=1,inplace=False).to_csv(logname, mode='a', index=False, header=0)

        if os.path.exists(shapname):
            if mode == 'gpd':
                log.crs = crs
                log.to_file(shapname, driver='ESRI Shapefile', mode='a')
        else:
            if mode == 'gpd':
                log.crs = crs
                log.to_file(shapname, driver='ESRI Shapefile', mode='w')

    @staticmethod
    def write_pd(Union_ex, index, lake_geometry,Img,mode='gpd', Method='SNIC_Kmean', Band=[0, 1, 3], WithOrigin=0, pd_dict=None,
                 Area_real=None, logname='log.csv', shapname='log.shp', calIoU=False,cal_resultArea=False,returnParms=False):

        if cal_resultArea:
            Area_ = Union_ex.area().divide(ee.Number(1000 * 1000)).getInfo()
        else:
            Area_ = False

        if calIoU:
            IoU = calculate_iou(Union_ex, lake_geometry).getInfo()
        else:
            IoU = False

        if mode == 'gpd':
            log = gpd.GeoDataFrame.from_features([Union_ex.getInfo()])
            log = log.assign(**{'Method': Method,
                                'Image':Img,
                                'Band': str(Band),
                                'WithOrigin': WithOrigin,
                                **pd_dict,
                                'Area_pre': [Area_],
                                'Area_real': [Area_real],
                                'IoU': IoU,
                                'index': index})
        else:
            log = pd.DataFrame({'Method': Method,
                                'Image': Img,
                                'Band': str(Band),
                                'WithOrigin': WithOrigin,
                                **pd_dict,
                                'Area_pre': [Area_],
                                'Area_real': [Area_real],
                                'IoU': IoU, },
                               index=[index])

        save_parms.save_log(log, mode=mode, logname=logname, shapname=shapname)
        if returnParms:
            return {'Method': Method,
                    'Image':Img,
                    'Band': str(Band),
                    'WithOrigin': WithOrigin,
                    **pd_dict,
                    'Area_pre': [Area_],
                    'Area_real': [Area_real],
                    'IoU': IoU,
                    'index': index}
        