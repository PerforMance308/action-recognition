import numpy as np
import scipy
import scipy.io
from time import time
import numpy.linalg as lin
import pickle
from sklearn.preprocessing import MinMaxScaler

Regression=0
Classifier=1

class Autoencoder:
    def __init__(self, elm_type, NumberofHiddenNeurons, C, kkkk, sn, name):
        self.elm_type = elm_type
        self.sn = sn
        self.kkkk = kkkk
        self.NumberofHiddenNeurons = NumberofHiddenNeurons
        self.C = C
        self.name = name
        self.fdafe=0
        self.fdafe1=0

    def picklefile(self, file, data):
        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def train(self, train_data, test_data):
        T = train_data[:,:1].T
        OrgT = T
        P = train_data[:,1:].T
        OrgP = P

        TVT = test_data[:,:1].T
        OrgTT = TVT
        TVP = test_data[:,1:].T

        NumberofTrainingData = P.shape[1]
        NumberofTestingData = TVP.shape[1]
        NumberofInputNeurons = P.shape[0]

        if self.elm_type != Regression:
            ls = np.concatenate((T, TVT), axis=1)
            sorted_target = np.sort(ls)
            label = []
            label.append(sorted_target[0][0])
            j = 1

            for i in range(NumberofTrainingData+NumberofTestingData):
                if sorted_target[0][i] != label[j-1]:
                    j += 1
                    label.append(sorted_target[0][i])

            number_class = j
            NumberofOutputNeurons = number_class

            temp_T = np.zeros([NumberofOutputNeurons, NumberofTrainingData])
            for i in range(NumberofTrainingData):
                for j in range(number_class):
                    if label[j] == T[0][i]:
                        break

                temp_T[j][i] = 1
            T = temp_T*2 - 1

            temp_TV_T = np.zeros([NumberofOutputNeurons, NumberofTestingData])
            for i in range(NumberofTestingData):
                for j in range(number_class):
                    if label[j] == TVT[0][i]:
                        break

                temp_TV_T[j][i] = 1
            TVT = temp_TV_T*2 - 1

        NumberofCategory = T.shape[0]
        start_train_time = time()
        saveT = T
        FTY = []
        TrainingAccuracy = []
        for subnetwork in range(self.sn):
            for j in range(self.kkkk):
                count = 1
                for nxh in range(count):
                    if j == 0:
                        BiasofHiddenNeurons1 = np.random.rand(self.NumberofHiddenNeurons, 1)
                        BiasofHiddenNeurons1 = scipy.linalg.orth(BiasofHiddenNeurons1)
                        InputWeight1 = np.random.rand(self.NumberofHiddenNeurons, NumberofInputNeurons) * 2 - 1
                
                        if self.NumberofHiddenNeurons > NumberofInputNeurons:
                            InputWeight1 = scipy.linalg.orth(InputWeight1)
                        else:
                            InputWeight1 = scipy.linalg.orth(InputWeight1.T).T

                        tempH = InputWeight1.dot(P)
                        tempH = tempH + BiasofHiddenNeurons1
                    else:
                        if nxh == 0:
                            My_PP = PP

                        x = np.float32(PP)
                        PP1= np.arcsin(PP)
                        PP1 = np.real(PP1)
                        P = P_save

                        m0 = np.eye(P.shape[0]) / self.C + P.dot(P.T)
                        n0 = P.dot(PP1.T)
                        InputWeight1 = lin.solve(m0,n0).T
                        fdafe = 0

                        tempH= InputWeight1.dot(P)
                        MyH = InputWeight1.dot(np.concatenate((P, TVP), axis = 1))
                        BB2 = np.sum(np.sum(tempH - PP1))
                        BBP = BB2 / PP1.shape[1]

                        tempH = (tempH.T - BBP.transpose()).T
                        MytempH = (MyH.T - BBP.transpose()).T

                    H = np.sin(tempH)

                    if j > 0:
                        MyH = np.sin(MytempH)
                        scaler = MinMaxScaler(feature_range=(-1,1))
                        scaler.fit(MyH)
                        MyH = scaler.transform(MyH)
                        H = MyH[:,0:NumberofTrainingData]
                        datainfo = {'YYM_H' : MyH}
                        scipy.io.savemat('./data/features/' + self.name+'feature_1.mat', datainfo)

                P_save = P
                P = H
                #FT = np.zeros((3, 17766))
                E1=T

                FT1 = []
                for i in range(2):
                    a = 0.0000001
                    Y2 = E1
                    scaler = MinMaxScaler(feature_range=(-1,1))
                    scaler.fit(Y2)
                    Y22 = scaler.transform(Y2)
                    Y2 = Y22
                    x = np.float32(Y2)
                    Y4 = np.arcsin(x)
                    Y4 = np.real(Y4)

                    if self.fdafe == 0:
                        m = np.eye(P.shape[0]) / self.C + P.dot(P.T)
                        n = P.dot(Y4.T)
                        g = lin.solve(m,n)
                        YYM = g

                        YJX = YYM.T.dot(P).T
                    else:
                        a1 = np.eye(YYM.shape[0]) / self.C + YYM.dot(YYM.T)
                        a2 = lin.solve(a1, YYM)
                        a3 = Y4.T.dot(a2.T)
                        PP = a3.T
                        YJX = PP.T.dot(YYM)

                    BB1 = Y4.shape
                    BB2 = (YJX - Y4.T).sum(axis = 0)
                    BB = BB2 / BB1[1]
                    BB = BB[0]
                    GXZ111 = (YJX.T - BB.transpose()).T
                    GXZ2 = np.sin(GXZ111.T)
                    FYY = scaler.inverse_transform(GXZ2).T
                    FT1.append(FYY.T)
                    E1 = E1-FT1[i]

                    end_train_time = time()
                    TrainingTime = end_train_time-start_train_time

                    if i==0:
                        FT = FT1[i]
                        self.fdafe = 1
                    else:
                        FT = FT + FT1[i]

                    if subnetwork == 0:
                        FTY.append(FT)
                    else:
                        FTY[subnetwork] = FTY[subnetwork] - FT
                    if self.elm_type == 1:
                        MissClassificationRate_Training = 0
                        for j in range(T.shape[1]):
                            x, label_index_expected = saveT[:, j].max(0), saveT[:, j].argmax(0)
                            x, label_index_actual = FTY[subnetwork][:, j].max(0), FTY[subnetwork][:, j].argmax(0)

                            if label_index_actual != label_index_expected:
                                MissClassificationRate_Training += 1

                        TrainingAccuracy.append(1 - MissClassificationRate_Training / T.shape[1])

                PP = PP + P
                min_max_scaler = MinMaxScaler(feature_range=(-0.99, 0.99))
                PP = min_max_scaler.fit_transform(PP.T).transpose()

            T = E1
            P = OrgP
            outputweight = YYM
            self.fdafe = 0

        i = 1

