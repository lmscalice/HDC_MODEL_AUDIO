import copy
import pickle
import librosa
import scipy
import scipy.io.wavfile
from scipy.fftpack import dct
from sklearn.preprocessing import MinMaxScaler
import os
import random
import argparse
import numpy as np

# Converts time domain signal to its MFCCs
def convertToMFCC2(sample_rate, signal):
    # Pre-Emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    frame_size = 0.025      # Frame Size: 25ms
    frame_stride = 0.01     # Stride: 10ms (means 15 ms overlap)
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Window
    frames *= np.hamming(frame_length)

    # Fourier-Transform and Power Spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # Filter Banks
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB  

    # Get Coefficients
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

    # Normalization
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    # MFCC-2
    band_range_percentages, band_sums, band_mean, band_std_dev = [], [], [], []
    num_frames = mfcc.shape[0]
    for y in range(mfcc.shape[1]):
        # Get band percentages for new set of classes between min and max
        class_tallies = [0] * 18
        max = np.amax(mfcc[:,y])
        min = np.amin(mfcc[:,y])
        difference = max-min
        step=difference/17 
        new_range = np.arange(min, max, step)

        for x in range(mfcc.shape[0]):
            for i in range(new_range.size-1, -1, -1):
                if (mfcc[x][y] > new_range[i]):
                    class_tallies[i]=class_tallies[i]+1
                    break
        for i in range(0, len(class_tallies)):
            class_tallies[i]=(class_tallies[i]/num_frames) *100
        band_range_percentages.append(class_tallies)

        # Get sum of values in band
        band_sums.append(np.sum(mfcc[:,y]))

        # Get mean of values in band
        band_mean.append(np.mean(mfcc[:,y]))

        # Get standard deviation of values in band
        band_std_dev.append(np.std(mfcc[:,y]))

    # Concatenate features for sample
    return np.concatenate((band_range_percentages, band_sums, band_mean, band_std_dev), axis=None)

# Normalizes data between -1 and 1
def normalizeData(data):
    return MinMaxScaler((-1,1)).fit_transform(data)

# Loading training and testing data
def dataLoader(filepath):
    num_features = 252
    Xtr = np.empty((0,num_features), int)
    Xts = np.empty((0,num_features), int)
    ytrain = []
    ytest = []
    for filename in os.listdir(filepath):
        first_split = filename.rsplit("_", 1)[1]
        audio_index = first_split.rsplit(".", 1)[0]
        classification = filename.split("_",1)[0]
        if int(audio_index) <= 4:
            ytest.append(classification)
            test_samples, test_samp_rate = librosa.load(filepath+filename, sr = None, mono =True, offset = 0.0, duration = None)
            mfccs = convertToMFCC2(test_samp_rate, test_samples)
            Xts = np.append(Xts, np.array([mfccs]), axis=0)
        else:
            ytrain.append(classification)
            test_samples, test_samp_rate = librosa.load(filepath+filename, sr = None, mono =True, offset = 0.0, duration = None)
            mfccs = convertToMFCC2(test_samp_rate, test_samples)
            Xtr = np.append(Xtr, np.array([mfccs]), axis=0)   
    Xtr= normalizeData(Xtr)
    Xts= normalizeData(Xts)
    ytr = np.array(ytrain)
    yts = np.array(ytest) 
    return Xtr, Xts, ytr, yts

# Binary search
def numToKey(value, levelList):
    if (value == levelList[-1]):
        return len(levelList)-2
    upperIndex = len(levelList) - 1
    lowerIndex = 0
    keyIndex = 0
    while (upperIndex > lowerIndex):
        keyIndex = int((upperIndex + lowerIndex)/2)
        if (levelList[keyIndex] <= value and levelList[keyIndex+1] > value):
            return keyIndex
        if (levelList[keyIndex] > value):
            upperIndex = keyIndex
            keyIndex = int((upperIndex + lowerIndex)/2)
        else:
            lowerIndex = keyIndex
            keyIndex = int((upperIndex + lowerIndex)/2)
    return keyIndex

def genLabel(dataset):
    label = dataset.reshape(-1).tolist()
    return label

def quantz(hdvector):
    HV = copy.deepcopy(hdvector)
    HV = np.array(HV)
    threshold = np.mean(HV)

    HV[HV > threshold] = 1
    HV[HV == threshold] = 0
    HV[HV < threshold] = -1
    return HV.tolist()

def getlevelList(totalLevel, minimum, maximum):
    levelList = []
    length = maximum - minimum
    gap = length / totalLevel
    for level in range(totalLevel):
        levelList.append(minimum + level*gap)
    levelList.append(maximum)
    return levelList

def HV_encoding(HDC, baseVector, levelVector, trainingData, testingData):
    HV_train, HV_test = [], []
    levelVector = copy.deepcopy(levelVector)
    dimension = HDC.dim
    print(trainingData.shape[0])
    for i in range(trainingData.shape[0]):
        trainData = trainingData[i, :]
        hdv = HDC.encoding(dimension, trainData, levelVector, baseVector)
        hdv = quantz(hdv)
        #hdv[hdv >= 0 ] = 1
        #hdv[hdv == 0] = 0
        #hdv[hdv < 0] = -1
        
        HV_train.append(hdv)
        print("running....")
        print(i)
    print("done")
    print(testingData.shape[0])
    print(testingData)
    for i in range(testingData.shape[0]):
        testData = testingData[i, :]
        print("about to run encode")
        hdv = HDC.encoding(dimension, testData, levelVector, baseVector)
        print("ran encode")
        hdv = quantz(hdv)
        print("running here")
        #hdv[hdv >= 0 ] = 1
        #hdv[hdv == 0] = 0
        #hdv[hdv < 0] = -1
        HV_test.append(hdv)
    return np.array(HV_train), np.array(HV_test)

def checkVector(classHVs, inputHV):
    guess = list(classHVs.keys())[0]
    maximum = np.NINF
    count, checklist = {} ,[]
    for key in classHVs.keys():
        count[key] = associateSearch(classHVs[key], inputHV)
        # inner_product(classHVs[key], inputHV)
        checklist.append([key, associateSearch(classHVs[key], inputHV)])
        if (count[key] > maximum):
            guess = key
            maximum = count[key]
    checklist = sorted(checklist, key = lambda x: x[1], reverse = -1)
    return guess, checklist

def associateSearch(HV1, HV2):
    return np.dot(HV1, HV2)/(np.linalg.norm(HV1) * np.linalg.norm(HV2) + 0.0)

### Save AM and Item Memory into file
def savemodel(am, baseVector, levelVector, levelList, fpath):
    f = open(fpath, 'wb')
    pickle.dump([am, baseVector, levelVector, levelList], f)
    f.close()
    return 0

def main(dimension, iteration, training, testing):
    # initializes the HDC object and its values
    HDC = HyperDimensionalComputing(dimension, totalPos = training[0].shape[1], totalLevel = 100, datatype = np.int16, buffer = [-1.0, 1.0], cuda = False)
    # separates the training images/labels and the testing images/labels
    trainingData, testingData, trainLabel, testLabel = training[0], testing[0], genLabel(training[1]), genLabel(testing[1])
    print("Running ...")
    classHV = dict([(x, np.array([0 for _ in range(dimension)])) for x in range(1, len(np.unique(testLabel)) + 1)])
    print("Still Running 1...")
    baseVector = HDC.genBaseVector(HDC.P, -1, HDC.dim)
    print("Still Running 2...")
    levelVector = HDC.genLevelVector(HDC.Q, -1, HDC.dim)
    print("Still Running 3...")
    HVector, HVector_test = HV_encoding(HDC, baseVector, levelVector, trainingData, testingData)
    print("Still Running 4...")
    classHVs = HDC.genClassHV(classHV, trainLabel , HVector)
    print("Still Running 5...")
    currWeight, currAcc = HDC.oneShotTraining(classHVs, HVector, trainLabel, HVector_test, testLabel,)
    print('One shot accuracy:', currAcc)
    print('-------- Retrain', iteration, 'epochs --------')
    currWeight, currAcc, bestWeight, bestAcc = HDC.retraining(currWeight, HVector, trainLabel , HVector_test, testLabel, iteration)
    
    ### save the assoc memory and item memory to a file
    fpath = './model_best'
    savemodel(bestWeight, baseVector, levelVector, HDC.levelList, fpath)
    return


#HDC model
class HyperDimensionalComputing(object):
    def __init__(self, dimension, totalPos, totalLevel, datatype, buffer, *string, cuda = False):
        self.P = totalPos                                                               # total number of features
        self.Q = totalLevel                                                             # number of levels in the buffer
        self.dim = dimension                                                            # Dimension of hyper-vector using in HDC model
        self.buffer = buffer                                                            # interval the feature values stay within
        self.datatype = datatype                                                        # data is of integer type
        self.levelList = getlevelList(totalLevel, self.buffer[0], self.buffer[1])       # gets the list of values in the buffer interval 
        
    def genBaseVector(self, totalPos, baseVal, dimension):
        D = dimension
        baseHVs = dict()
        indexVector = range(D)
        change = int(D/2)
        for level in range(totalPos):
            name = level
            base = np.full(D, baseVal)
            toOne = np.random.permutation(indexVector)[:change]  
            for index in toOne:
                base[index] = 1
            baseHVs[name] = copy.deepcopy(base)     
        return baseHVs

    def genLevelVector(self, totalLevel, baseVal, dimension):
        D = dimension
        levelHVs = dict()
        indexVector = range(D)
        nextLevel = int((D/2/totalLevel))
        change = int(D/2)
        for level in range(totalLevel):
            name = level
            if(level == 0):
                base = np.full(D, baseVal)
                toOne = np.random.permutation(indexVector)[:change]
            else:
                toOne = np.random.permutation(indexVector)[:nextLevel]
            for index in toOne:
                base[index] = base[index] * -1
            levelHVs[name] = copy.deepcopy(base)
        return levelHVs

    def encoding(self, dimension, label, levelHVs, baseHVs):
        HDVector = np.zeros(dimension, dtype = self.datatype)
        for keyVal in range(len(label)):
            key = numToKey(label[keyVal], self.levelList)
            baseHV = baseHVs[keyVal]
            levelHV = levelHVs[key]
            HDVector = HDVector + (baseHV * levelHV)
        return HDVector

    def genClassHV(self, classHV, inputLabels, inputHVs):
        #This creates a dict with no duplicates
        classHVs = copy.deepcopy(classHV)
        for i in range(len(inputLabels)):
            name = inputLabels[i]
            classHVs[name] = np.array(classHVs[name]) + np.array(inputHVs[i])
        return classHVs

    def oneShotTraining(self, classHVs, trainHVs, trainLabels, testHVs, testLabels,):
        retClassHVs = copy.deepcopy(classHVs)
        currAcc = self.test(retClassHVs, testHVs, testLabels)
        for index in range(len(trainLabels)):
            predict, dis_checklist = checkVector(retClassHVs, trainHVs[index])
            if not (trainLabels[index] == predict):
                retClassHVs[predict] = retClassHVs[predict] - trainHVs[index]
                retClassHVs[trainLabels[index]] = retClassHVs[trainLabels[index]] + trainHVs[index]
        return retClassHVs, currAcc

    def retraining (self, classHVs, trainHVs, trainLabels, testHVs, testLabels, n_iteration):
        bestWeight, bestacc = dict(), 0
        currClassHV = copy.deepcopy(classHVs)
        for i in range(n_iteration):
            currClassHV, _ = self.oneShotTraining(currClassHV, trainHVs, trainLabels, testHVs, testLabels,)
            currAcc = self.test(currClassHV, testHVs, testLabels)
            print('Epoch', i, 'accuracy:', currAcc)
            if currAcc > bestacc:
                bestAcc = currAcc
                bestWeight = currClassHV
        return currClassHV, currAcc, bestWeight, bestAcc

    def test(self, classHVs, testHVs, testLabels):
        correct = 0
        for index in range(len(testHVs)):
            predict, checklist = checkVector(classHVs, testHVs[index])
            if (testLabels[index] == predict):
                correct += 1
        accuracy = (correct / len(testLabels)) * 100
        return accuracy
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--app_', metavar='N', type=str)
    parser.add_argument('--iter_', metavar='N', type=int)
    parser.add_argument('--dimension_', metavar='N', type=int)

    args = parser.parse_args()
    Xtr, Xts, ytr, yts = dataLoader(args.app_)
    main(args.dimension_, args.iter_, [Xtr, ytr], [Xts, yts])
