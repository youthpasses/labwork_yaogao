# coding:utf-8
# @makai
# 17/2/16

import gdal
from gdalconst import *
import numpy as np
import os
import Image
import struct
import pandas as pd

DATA_DIR = '../data/'
LABEL_PATH = '../data/label.txt'
ORIGIN_X = 2195
ORIGIN_Y = 3773
WIDTH = 5128
HEIGHT = 3776

#df = pd.read_csv(LABEL_PATH, header=None, names=['1', 'label', '3', '4'])
#df.drop(['1', '3', '4'], axis=1, inplace=True)

# df = pd.read_table(LABEL_PATH, header=None, names=['1', 'label'])
# print df.info()


f2 = open('../data/label.txt', 'r')
lines = f2.readlines()
line = lines[0]
labels_all = [int(x) for x in line.split(',')]

def getIndexWithLocation(x, y):
	return y * WIDTH + x

def getLabelsWithArea(luX, luY, rbX, rbY, labels_all):
	x1 = luX - ORIGIN_X
	y1 = luY - ORIGIN_Y
	x2 = rbX - ORIGIN_X
	y2 = rbY - ORIGIN_Y
	#print x1, y1, x2, y2
	areaWidth = rbX - luX + 1
	areaHeight = rbY - luY + 1
	#print areaWidth, areaHeight
	labels = []
	for i in xrange(0, areaHeight):
		y = y1 + i
		index = getIndexWithLocation(x1, y)
		labels += labels_all[index: index + areaWidth]
	return labels

# label1 = getLabelsWithArea(2195,3773, 2214, 3773, labels_all)
# print label1, len(label1)

def getDataFromTifWithArea(luX, luY, rbX, rbY, filepath):
	tf = gdal.Open(filepath, GA_ReadOnly)
	if not tf:
		print 'File open fails.'
	band = tf.GetRasterBand(1)
	# data = band.ReadAsArray(2195, 3773,10,10)
	data1 = np.array(band.ReadAsArray(luX, luY, rbX - luX + 1, rbY - luY + 1))
	data1 = np.reshape(data1, (-1))
	return data1
# getDataFromTifWithArea(2195, 3773, 2195 + 10, 3773 + 10, DATA_DIR + 'data4.tif')




def getTrainDataFrame():
	areas = [[3621, 5316, 3850, 5449], 
			[4605, 6733, 4804, 6947], 
			[2721, 7180, 2826, 7279],
			[6000, 5309, 6149, 5468],
			[5700, 5630, 5859, 5789]]
	return getDataFrame(areas)


def getTestDataFrame():
	areas = [[2195, 3773, 7322, 7548]]
	return getDataFrame(areas)


def getDataFrame(areas):
	data = []
	label = []
	for area in areas:
		print area
		luX = area[0]
		luY = area[1]
		rbX = area[2]
		rbY = area[3]
		num = (rbX - luX + 1) * (rbY - luY + 1)
		data1 = [[] for x in xrange(0, num)]
		label1 = getLabelsWithArea(luX, luY, rbX, rbY, labels_all)
		for i in xrange(1, 8):
			filepath = DATA_DIR + 'data' + str(i) + '.tif'
			# print area, filepath
			data2 = getDataFromTifWithArea(luX, luY, rbX, rbY, filepath)
			temp = zip(data1, data2)
			data1 = [x[0] + [x[1]] for x in temp]
		data.extend(data1)
		label.extend(label1)
	# analyse the count of every class
	# label_set = set(label)
	# count_dic = {}
	# for l in label_set:
	# 	count_dic[l] = label.count(l)
	# print count_dic
	# {1: 20388, 2: 16904, 3: 5069, 5: 13892, 7: 19073, 8: 13360, 9: 11399, 10: 7687, 11: 18776, 12: 7472}
	# 11 -> 4, 12 -> 6
	label = [int(x) - 1 for x in label]
	data = np.array(data).astype(np.float32)
	label = np.array(label)
	label[label == 10] = 3
	label[label == 11] = 5
	print 'data.shape: ', data.shape
	print 'label.shape: ', label.shape
	dic = {'label': label}
	dic['f1'] = data.T[0]
	dic['f2'] = data.T[1]
	dic['f3'] = data.T[2]
	dic['f4'] = data.T[3]
	dic['f5'] = data.T[4]
	dic['f6'] = data.T[5]
	dic['f7'] = data.T[6]
	df = pd.DataFrame(data=dic, columns=dic.keys())
	# print 'df.head() : \n', df.head()
	# print 'df.tail() : \n', df.tail()
	# print 'df.info() : \n', df.info()
	return df






























def readTiffFile(tiffPath):
	tf = gdal.Open(tiffPath, GA_ReadOnly)
	if not tf:
		print 'File open fails.'
		return
	#print 'Driver:', tf.GetDriver().ShortName, '/', tf.GetDriver().LongName
	#print 'Size is ', tf.RasterXSize,'x',tf.RasterYSize, 'x',tf.RasterCount
	#print 'Projection is ',tf.GetProjection()
	#geotransform = tf.GetGeoTransform()
	#print 'geotransform: ', geotransform
	#if geotransform:
	#    print 'Origin = (',geotransform[0], ',',geotransform[3],')'
	#    print 'Pixel Size = (',geotransform[1], ',',geotransform[5],')'
	#else:
	#	print 'geotransform is None.'
	band = tf.GetRasterBand(1)
	#print 'Band Type=',gdal.GetDataTypeName(band.DataType)
	#min = band.GetMinimum()
	#print min
	#max = band.GetMaximum()
	#if min is None or max is None:
	#    (min,max) = band.ComputeRasterMinMax(1)
	#print 'Min=%.3f, Max=%.3f' % (min,max)
	#if band.GetOverviewCount() > 0:
	#    print 'Band has ', band.GetOverviewCount(), ' overviews.'
	#if not band.GetRasterColorTable() is None:
	#    print 'Band has a color table with ', \
	    # band.GetRasterColorTable().GetCount(), ' entries.'
	#print 'band.XSize = ', band.XSize
	#print 'band.YSize = ', band.YSize
	# scanline = band.ReadRaster( 2195, 3773, 100, 1, 100, 1, GDT_Float32 )
	# print scanline[:10]
	# tuple_of_floats = struct.unpack('f' * 100 * 100 * 4, scanline)
	# print len(tuple_of_floats)
	#print tf.ModelPixelScaleTag
	data = band.ReadAsArray(2195, 3773,10,10)
	print data.shape

# readTiffFile(DATA_DIR + 'data4.tif')