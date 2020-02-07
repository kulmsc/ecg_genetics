import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn import metrics
from random import randint
import random
import numpy as np
from os import listdir
import random
import pdb
import keras
import sklearn.metrics as sklm

try:
    import matplotlib.pyplot as plt
except:
    print('no matplotlib')
try:
    from sklearn.decomposition import PCA
except: 
    print('no sklearn')
try:
    import pandas as pd
except: 
    print('no pandas')



class Models():
    """
    A Class for creating all ML models
    """

    def __init__(self, model_name='autoencoder',
                 slices=3, num_channels=12, test_proportion=.8, 
                 ecg_time_samples=5000):
        self.slices = slices
        self.num_channels = num_channels
        self.test_proportion = test_proportion

        self.normal_training = np.array([])
        self.normal_testing = np.array([])
        self.afib_data = np.array([])
        self.current_model = []
        self.encoder = []
        self.norm_predict = []
        self.history = []

        #scott added
        self.normal_training_names = []
        self.normal_testing_names = []
        self.afib_data_names = []
        self.normal_training_genetic = np.array([])
        self.normal_testing_genetic = np.array([])
        self.afib_data_genetic = np.array([])

        self.signal_length = int(ecg_time_samples/slices)
        if not ((self.signal_length % 2) == 0):
            self.signal_length -= 1

        self.input_shape = Input(shape=(self.signal_length,num_channels))

    def load_data(self, path_to_data, genetic_path, num_files=100, rhythm_type='normal', genetic_with_ecg=True):
        file_names = listdir(path_to_data)
       
        #alter the number of files so all ecg also have genetic data
        if 'withEcg' in genetic_path:
            genetic_names = np.genfromtxt(genetic_path, skip_header = 1, missing_values = "NA", usecols = 1)
            file_names = [x for x in file_names if any(int(x.split('.')[0]) == genetic_names)]

        if len(file_names) > num_files:
            file_names = file_names[0:num_files]

        input_data = np.zeros((len(file_names)*self.slices,
                               self.signal_length,
                               self.num_channels))

        index = 0
        t_span = self.signal_length
        for file_name in file_names:
            current_path = f'{path_to_data}/{file_name}'
            current_signal = np.load(current_path)
            current_signal = self._norm_signal_channels(current_signal)
            for num in range(self.slices):
                start_index = num * t_span
                end_index = (num+1)*t_span
                input_data[index, :, :] = current_signal[start_index:end_index,
                                                         0:self.num_channels]
                index+=1



        num_training = round(num_files * self.slices * self.test_proportion)
        if rhythm_type == 'normal':
            self.normal_training = input_data[0:num_training, :, :]
            self.normal_testing = input_data[num_training:, :, :]
            #scott added
            self.normal_training_names = file_names[0:num_training]
            self.normal_testing_names = file_names[num_training:]
        if rhythm_type == 'afib':
            self.afib_data = input_data
            #scott added
            self.afib_data_names = file_names

    #function scott added
    def load_genetic_data(self, genetic_path):
        all_genetic = np.genfromtxt(genetic_path, skip_header = 1, missing_values = "NA")
        sample_info = all_genetic[:,0:6]
        genetic_info = all_genetic[:,6:]

        #this is mean imputation, could also do median, or just set all nans to 0
        for i in range(genetic_info.shape[1]):
            genetic_info[np.isnan(genetic_info[:,i]), i] = np.mean(genetic_info[~np.isnan(genetic_info[:,i]), i])

        #pdb.set_trace()
        #set up the holding matricies
        if 'withEcg' in genetic_path:
            self.afib_data_genetic = np.zeros([self.afib_data.shape[0], int(genetic_info.shape[1]/2)])        
            self.normal_training_genetic = np.zeros([self.normal_training.shape[0], int(genetic_info.shape[1]/2)])
            self.normal_testing_genetic = np.zeros([self.normal_testing.shape[0], int(genetic_info.shape[1]/2)])

            #afib
            for i, file_name in enumerate(self.afib_data_names):
                fam_name = file_name.split('.')[0]
                sub_genetic = genetic_info[sample_info[:,0] == int(fam_name),:]
                self.afib_data_genetic[i,:] = sub_genetic[0, range(0, sub_genetic.shape[1], 2)]

            #normal - training
            for i, file_name in enumerate(self.normal_training_names):
                fam_name = file_name.split('.')[0]
                sub_genetic = genetic_info[sample_info[:,0] == int(fam_name),:]
                self.normal_training_genetic[i,:] = sub_genetic[0, range(0, sub_genetic.shape[1], 2)]

            #normal - testing
            for i, file_name in enumerate(self.normal_testing_names):
                fam_name = file_name.split('.')[0]
                sub_genetic = genetic_info[sample_info[:,0] == int(fam_name),:]
                self.normal_testing_genetic[i,:] = sub_genetic[0, range(0, sub_genetic.shape[1], 2)]

            #pdb.set_trace()
        else:
            pheno = np.loadtxt("/home/kulmsc/athena/ecg_kulm/data/genetic/all.pheno")
            num_train = int(sum(pheno==0)*self.test_proportion)
            num_test = sum(pheno==0) - num_train
            self.afib_data_genetic = np.zeros([sum(pheno==1), int(genetic_info.shape[1]/2)])
            self.afib_fams = ['x'] * self.afib_data_genetic.shape[0]
            self.normal_training_genetic = np.zeros([num_train, int(genetic_info.shape[1]/2)])
            self.normal_training_fams = ['x'] * self.normal_training_genetic.shape[0]
            self.normal_testing_genetic = np.zeros([num_test, int(genetic_info.shape[1]/2)])
            self.normal_testing_fams = ['x'] * self.normal_testing_genetic.shape[0]

            #afib
            for i, ind in enumerate(np.where(pheno==1)[0]):
                print(i)
                self.afib_fams[i] = sample_info[ind,0]
                sub_genetic = genetic_info[ind,:]
                self.afib_data_genetic[i,:] = [sub_genetic[j] for j in range(0, len(sub_genetic), 2)]

            #normal - training
            for i, ind in enumerate(np.where(pheno==0)[0][0:self.normal_training_genetic.shape[0]]):
                self.normal_training_fams[i] = sample_info[ind,0]
                sub_genetic = genetic_info[ind,:]
                self.normal_training_genetic[i,:] = [sub_genetic[j] for j in range(0, len(sub_genetic), 2)]

            #normal - testing
            for i, ind in enumerate(np.where(pheno==0)[0][(self.normal_training_genetic.shape[0]+1):]):
                self.normal_testing_fams[i] = sample_info[ind,0]
                sub_genetic = genetic_info[ind,:]
                self.normal_testing_genetic[i,:] = [sub_genetic[j] for j in range(0, len(sub_genetic), 2)]

        #dont think that I need to store most of this data to self
        mainPath = "/".join(genetic_path.split("/")[1:-1])
        self.genetic_stats = np.loadtxt('/'+mainPath+"/afibStats")                     #from afibSS, same order as bim
        self.genetic_alleles = np.loadtxt('/'+mainPath+"/afibAlleles", dtype = 'str')  #from afibSS, same order as bim
        genetic_ref = np.loadtxt('/'+mainPath+"/afibRef", dtype = 'str')               #from the imputed qc
        genetic_rsids = np.loadtxt('/'+mainPath+"/afibRsids", dtype = 'str')           #from afibSS, same order as bim

        #want to convert the betas
        bad_collector = []
        for i in range(genetic_ref.shape[0]):
            refIndex = np.where(genetic_ref[:,0] == genetic_rsids[i])[0][0]
            print("\n new")
            print(genetic_ref[refIndex,:])
            print(genetic_rsids[i])
            print(self.genetic_alleles[i,:])

            if genetic_ref[refIndex,1] == self.genetic_alleles[i,0].upper() and genetic_ref[refIndex,2] == self.genetic_alleles[i,1].upper():
                pass
            elif genetic_ref[refIndex,1] == self.genetic_alleles[i,1].upper() and genetic_ref[refIndex,2] == self.genetic_alleles[i,0].upper():
                self.genetic_stats[i,1] = -self.genetic_stats[i,1]
            elif genetic_ref[refIndex,1] == self.genetic_alleles[i,0].upper() and genetic_ref[refIndex,2] != self.genetic_alleles[i,1].upper():
                bad_collector.append(i)
            elif genetic_ref[refIndex,1] == self.genetic_alleles[i,1].upper() and genetic_ref[refIndex,2] != self.genetic_alleles[i,0].upper():
                bad_collector.append(i)
            elif genetic_ref[refIndex,2] == self.genetic_alleles[i,1].upper() and genetic_ref[refIndex,1] != self.genetic_alleles[i,0].upper():
                bad_collector.append(i)
            elif genetic_ref[refIndex,2] == self.genetic_alleles[i,0].upper() and genetic_ref[refIndex,1] != self.genetic_alleles[i,1].upper():
                bad_collector.append(i)
            else:
                print(i)
                raise Exception("something odd is happening")

        if len(bad_collector) > 0:
            raise Exception("there are bad snps being analyzed")
            self.bad_collector = bad_collector



    def _norm_signal_channels(self, signal_all_channels):
        return np.apply_along_axis(self._norm_column, 0, signal_all_channels)

    def _norm_column(self, signal):
        if signal.min() > 0:
            positive_signal = signal - signal.min()
        else:
            positive_signal = signal + np.abs(signal.min())
        normalized_signal = positive_signal / positive_signal.max()
        return normalized_signal

    def get_genetic_nn(self, layer1_nodes, layer2_nodes=0, incStat=True):
        input_shape = self.afib_data_genetic.shape
        #should check if the arrays contain nan before multiplication
        #pdb.set_trace()

        #assuming going with multiple option
        if incStat:
            self.normal_testing_genetic = np.multiply(self.normal_testing_genetic, self.genetic_stats[:,1])
            self.normal_training_genetic = np.multiply(self.normal_training_genetic, self.genetic_stats[:,1])
            self.afib_data_genetic = np.multiply(self.afib_data_genetic, self.genetic_stats[:,1])

        #dense
        # to add regularization add as an argument: kernel_regularizer=regularizers.l1(0.5)
        model = Sequential()
        model.add(Dense(layer1_nodes, input_dim = input_shape[1], activation = 'relu'))
        model.add(Dropout(0.5))
        if layer2_nodes != 0:
            model.add(Dense(layer2_nodes, activation = 'relu'))
        model.add(Dense(1, activation = 'sigmoid'))

        #compile
        myAdam = keras.optimizers.Adam(lr = 0.00001)
        model.compile(loss = 'binary_crossentropy', optimizer = myAdam, metrics = ['binary_accuracy'])
        print(model.summary())
        self.genetic_model = model
        #self.genetic_orig_weights = self.genetic_model.get_weights()

    def cv_genetic_nn(self):
        print("start cv_genetic_nn")
        all_controls = np.vstack((self.normal_training_genetic, self.normal_testing_genetic))
        all_cases = self.afib_data_genetic

        pdb.set_trace()
        control_prop = 0.8
        control_inds = np.concatenate([np.repeat(1, round(all_controls.shape[0] * control_prop)), \
                          np.repeat(2, (all_controls.shape[0] - round(all_controls.shape[0] * control_prop)))])
        case_inds = np.concatenate([np.repeat(1, round(all_cases.shape[0] * control_prop)), \
                          np.repeat(2, (all_cases.shape[0] - round(all_cases.shape[0] * control_prop)))])

        np.random.shuffle(control_inds)
        np.random.shuffle(case_inds)

        #may need to normalize somehow
        #self.genetic_acc = [i for i in range(3)]
        #self.genetic_auc = [i for i in range(3)]
        #self.genetic_pred_yes = [i for i in range(3)]
        #self.genetic_pred_no = [i for i in range(3)]


        sampleOpt = "doNothing"
        i=1 #to get x_train, ...
        if sampleOpt == "doNothing":
            x_train = np.vstack((all_controls[control_inds != i+1], all_cases[case_inds != i+1])) 
            y_train = np.hstack((np.zeros(np.sum(control_inds != i+1)), np.ones(np.sum(case_inds != i+1))))
            x_test = np.vstack((all_controls[control_inds == i+1], all_cases[case_inds == i+1]))
            y_test = np.hstack((np.zeros(np.sum(control_inds == i+1)), np.ones(np.sum(case_inds == i+1))))
        elif sampleOpt == "down":
            downTrain = sum(case_inds != i+1)
            downTest = sum(case_inds == i+1)
            fullControl = all_controls[control_inds != i+1]
            downControl = fullControl[np.random.choice(np.arange(fullControl.shape[0]), downTrain), :]
            x_train = np.vstack((downControl, all_cases[case_inds != i+1]))
            y_train = np.hstack((np.zeros(downTrain), np.ones(np.sum(case_inds != i+1))))
            fullControl = all_controls[control_inds == i+1]
            downControl = fullControl[np.random.choice(np.arange(fullControl.shape[0]), downTest), :]
            x_test = np.vstack((downControl, all_cases[case_inds == i+1]))
            y_test = np.hstack((np.zeros(downTest), np.ones(np.sum(case_inds == i+1))))
        elif sampleOpt == "up":
            upTrain = sum(control_inds != i+1)
            upTest = sum(control_inds == i+1)
            fullCasePull = all_cases[case_inds != i+1]
            fullCase = np.zeros((upTrain, all_controls.shape[1]))
            for j in range(fullCase.shape[0]):
                fullCase[j,:] = fullCasePull[randint(0,(fullCasePull.shape[0]-1)),:]
            x_train = np.vstack((all_controls[control_inds != i+1], fullCase))
            y_train = np.hstack((np.zeros(np.sum(control_inds != i+1)), np.ones(upTrain)))
            fullCasePull = all_cases[case_inds == i+1]
            fullCase = np.zeros((upTest, all_controls.shape[1]))
            for j in range(fullCase.shape[0]):
                fullCase[j,:] = fullCasePull[randint(0,(fullCasePull.shape[0]-1)),:]
            x_test = np.vstack((all_controls[control_inds == i+1], fullCase))
            y_test = np.hstack((np.zeros(np.sum(control_inds == i+1)), np.ones(upTest)))
        elif sampleOpt == "alter":
            #should do what is above but for each column (SNP) get the AF and then for a new person rbinom out whether they have the SNP
            upTrain = sum(control_inds != i+1)
            upTest = sum(control_inds == i+1)
            probOneInCase = np.zeros(all_controls.shape[1])
            numPeople = all_controls.shape[0]
            for j in range(probOneInCase.shape[0]):
                probOneInCase[j] = sum(all_cases[:,j] == 1)/numPeople

            addCase = np.zeros((upTrain - sum(case_inds != i+1), all_controls.shape[1]))
            for j in range(addCase.shape[1]):
                addCase[:,j] = [ int(random.random() < probOneInCase[j]) for k in range(upTrain - sum(case_inds != i+1)) ]
            x_train = np.vstack((all_controls[control_inds != i+1], all_cases[case_inds != i+1], addCase))
            y_train = np.hstack((np.zeros(np.sum(control_inds != i+1)), np.ones(upTrain)))


        pickMethod = "nn"
        comparePrs = True
        if pickMethod == "logreg":
            logreg = LogisticRegression(class_weight = 'balanced')
            self.logreg_betas = np.array([0.0 for i in range(x_train.shape[1])])
            for j in range(x_train.shape[1]):
                fitMod = logreg.fit(np.expand_dims(x_train[:,j],axis=1), y_train)
                self.logreg_betas[j] = fitMod.coef_[0][0]
            #np.corrcoef(self.genetic_stats[:,1], self.logreg_betas)
            prs_train = np.sum(x_train * self.logreg_betas, axis = 1)
            prs_test = np.sum(x_test * self.logreg_betas, axis = 1)
            totalMod = logreg.fit(np.expand_dims(prs_train,axis=1), y_train)
            probs = totalMod.predict_proba(np.expand_dims(prs_test,axis=1))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, probs[:,1])
            self.genetic_auc = metrics.auc(fpr, tpr)

            #tprDiff = tpr-np.linspace(0,1,len(tpr))
            #bestThresh = thresholds[np.where(tprDiff == max(tprDiff))]
            bestCalls = (probs[:,1] > 0.5)*1
            self.genetic_acc = np.sum(bestCalls == y_test)/len(y_test)

        elif pickMethod == "prs" or comparePrs:
            logreg = LogisticRegression(class_weight = 'balanced')
            prs_train = np.sum(x_train * self.genetic_stats[:,1], axis = 1)
            prs_test = np.sum(x_test * self.genetic_stats[:,1], axis = 1)
            if pickMethod == "prs":
                totalMod = logreg.fit(np.expand_dims(prs_train,axis=1), y_train)
                probs = totalMod.predict_proba(np.expand_dims(prs_test,axis=1))
                fpr, tpr, thresholds = metrics.roc_curve(y_test, probs[:,1])
                self.genetic_auc = metrics.auc(fpr, tpr)

                bestCalls = (probs[:,1] > 0.5)*1
                self.genetic_acc = np.sum(bestCalls == y_test)/len(y_test)

             

        if pickMethod == "nn":
            #class Metrics(keras.callbacks.Callback):
            #    def on_train_begin(self, logs={}):
            #        self.auc = []
            #    def on_epoch_end(self, epoch, logs={}):
            #        score = np.asarray(self.model.predict(self.validation_data[0]))
            #        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
            #        targ = self.validation_data[1]
            #        self.auc.append(sklm.roc_auc_score(targ, score))
            #        return 

            toshuf = np.arange(x_train.shape[0])
            np.random.shuffle(toshuf)
            x_train = x_train[toshuf,:]
            y_train = y_train[toshuf]
            toshuf = np.arange(x_test.shape[0])
            np.random.shuffle(toshuf)
            x_test = x_test[toshuf,:]
            y_test = y_test[toshuf]
            #self.genetic_model.set_weights(self.genetic_orig_weights)
            class_weight = {0:sum(y_train==1)/y_train.shape[0], 1:sum(y_train==0)/y_train.shape[0]}
            pdb.set_trace()
            #external_mod = self.genetic_model.fit(x_train, y_train, epochs = 150, batch_size = 32, class_weight = class_weight, validation_split = 0.8, callbacks=[metrics])
            external_mod = self.genetic_model.fit(x_train, y_train, epochs = 150, batch_size = 32, class_weight = class_weight, validation_split = 0.8)
            self.genetic_acc = self.genetic_model.evaluate(x_test, y_test)[1]
            probs = self.genetic_model.predict(x_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, probs[:,0])
            self.genetic_auc = metrics.auc(fpr, tpr)

            plotIt = True
            if plotIt:
                import matplotlib.pyplot as plt
                plt.switch_backend('agg')
                plt.figure()
                plt.subplot(211)
                plt.plot(external_mod.history['val_loss'])
                plt.plot(external_mod.history['loss'])
                plt.legend(['Test', 'Train'], loc='upper left')
                plt.ylabel('Loss')
                plt.title("Validation ACC: "+str(self.genetic_acc)[0:5])
                plt.suptitle("Validation AUC: "+str(self.genetic_auc)[0:5])

                plt.subplot(212)
                plt.plot(external_mod.history['val_binary_accuracy'])
                plt.plot(external_mod.history['binary_accuracy'])
                plt.legend(['Test', 'Train'], loc='lower right')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')

                plt.savefig('nn_epochs.png')

                if comparePrs:
                    plt.figure()
                    plt.scatter(probs, prs_test)
                    plt.ylabel('NN Probs')
                    plt.xlabel('PRS Test')
                    plt.savefig('nn_vs_prs.png')

        print("DONE")
        pdb.set_trace()

    def get_100x_autoencoder(self):
        input_shape = self.input_shape

        #encoder
        #1
        conv_e1 = Conv1D(32, 20, activation='relu', padding='same')(
            input_shape)
        pool_e1 = MaxPooling1D(2, padding='same')(conv_e1)

        #2
        conv_e2 = Conv1D(64, 4, activation='relu', padding='same')(
            pool_e1)
        conv_e2_2 = Conv1D(64, 4, activation='relu', padding='same')(
            conv_e2)
        pool_e2 = MaxPooling1D(2, padding='same')(conv_e2_2)

        #3
        conv_e3 = Conv1D(32, 4, activation='relu', padding='same')(
            pool_e2)
        conv_e3_2 = Conv1D(32, 4, activation='relu', padding='same')(
            conv_e3)
        pool_e3 = MaxPooling1D(2, padding='same')(conv_e3_2)

        #4
        conv_e4 = Conv1D(32, 4, activation='relu', padding='same')(
            pool_e3)
        conv_e4_2 = Conv1D(16, 4, activation='relu', padding='same')(
            conv_e4)
        pool_e4 = MaxPooling1D(2, padding='same')(conv_e4_2)

        #5
        conv_e5 = Conv1D(16, 4, activation='relu', padding='same')(
            pool_e4)
        conv_e5_2 = Conv1D(8, 4, activation='relu', padding='same')(
            conv_e5)
        pool_e5 = MaxPooling1D(2, padding='same')(conv_e5_2)

        #6
        conv_e6 = Conv1D(4, 4, activation='relu', padding='same')(
            pool_e5)
        conv_e6_2 = Conv1D(3, 4, activation='relu', padding='same')(
            conv_e6)
        pool_e6 = MaxPooling1D(2, padding='same')(conv_e6_2)

        #7
        #conv_e7 = Conv1D(4, 4, activation='relu', padding='same')(
        #    pool_e6)
        #conv_e7_2 = Conv1D(2, 4, activation='relu', padding='same')(
        #    conv_e7)
        #pool_e7 = MaxPooling1D(2, padding='same')(
        #    conv_e7_2)


        #7
        #conv_d7 = Conv1D(2, 4, activation='relu', padding='same')(pool_e7)
        #conv_d7_2 = Conv1D(4, 4, activation='relu', padding='same')(conv_d7)
        #up7 = UpSampling1D(2)(conv_d7_2)

        #6
        conv_d6 = Conv1D(3, 4, activation='relu', padding='same')(pool_e6)
        conv_d6_2 = Conv1D(4, 4, activation='relu', padding='same')(conv_d6)
        up6 = UpSampling1D(2)(conv_d6_2)

        #5
        conv_d5 = Conv1D(8, 4, activation='relu', padding='same')(up6)
        conv_d5_2 = Conv1D(16, 4, activation='relu', padding='same')(conv_d5)
        up5 = UpSampling1D(2)(conv_d5_2)


        #4
        conv_d4 = Conv1D(16, 4, activation='relu', padding='same')(up5)
        conv_d4_2 = Conv1D(32, 4, activation='relu', padding='same')(conv_d4)
        up4 = UpSampling1D(2)(conv_d4_2)


        #3
        conv_d3 = Conv1D(32, 4, activation='relu', padding='same')(up4)
        conv_d3_2 = Conv1D(32, 4, activation='relu', padding='same')(conv_d3)
        up3 = UpSampling1D(2)(conv_d3_2)

        #2 
        conv_d2_2 = Conv1D(64, 4, activation='relu', padding='same')(up3)
        conv_d2 = Conv1D(32, 30, activation='relu', padding='same')(conv_d2_2)
        up2 = UpSampling1D(2)(conv_d2)

        #1
        conv_d1 = Conv1D(32, 4, activation='relu', padding='same')(up2)
        up1 = UpSampling1D(2)(conv_d1)

        #out
        rec_signal = Dense(12, activation='sigmoid')(up1)

        if rec_signal.shape[1].value != self.signal_length:
            crop_length = int((rec_signal.shape[1].value - self.signal_length)/2)
            rec_signal = Cropping1D(crop_length)(rec_signal)

        autoencoder = Model(input=input_shape, output=rec_signal)
        autoencoder.compile(optimizer='adam', loss='mse')

        self.current_model = autoencoder

    def get_50x_autoencoder(self):
        input_shape = self.input_shape

        #encoder
        #1
        conv_e1 = Conv1D(32, 20, activation='relu', padding='same')(
            input_shape)
        pool_e1 = MaxPooling1D(2, padding='same')(conv_e1)

        #2
        conv_e2 = Conv1D(64, 4, activation='relu', padding='same')(
            pool_e1)
        conv_e2_2 = Conv1D(64, 4, activation='relu', padding='same')(
            conv_e2)
        pool_e2 = MaxPooling1D(2, padding='same')(conv_e2_2)

        #3
        conv_e3 = Conv1D(32, 4, activation='relu', padding='same')(
            pool_e2)
        conv_e3_2 = Conv1D(32, 4, activation='relu', padding='same')(
            conv_e3)
        pool_e3 = MaxPooling1D(2, padding='same')(conv_e3_2)

        #4
        conv_e4 = Conv1D(32, 4, activation='relu', padding='same')(
            pool_e3)
        conv_e4_2 = Conv1D(16, 4, activation='relu', padding='same')(
            conv_e4)
        pool_e4 = MaxPooling1D(2, padding='same')(conv_e4_2)

        #5
        conv_e5 = Conv1D(16, 4, activation='relu', padding='same')(
            pool_e4)
        conv_e5_2 = Conv1D(8, 4, activation='relu', padding='same')(
            conv_e5)
        pool_e5 = MaxPooling1D(2, padding='same')(conv_e5_2)


        #5
        conv_d5 = Conv1D(8, 4, activation='relu', padding='same')(pool_e5)
        conv_d5_2 = Conv1D(16, 4, activation='relu', padding='same')(conv_d5)
        up5 = UpSampling1D(2)(conv_d5_2)


        #4
        conv_d4 = Conv1D(16, 4, activation='relu', padding='same')(up5)
        conv_d4_2 = Conv1D(32, 4, activation='relu', padding='same')(conv_d4)
        up4 = UpSampling1D(2)(conv_d4_2)


        #3
        conv_d3 = Conv1D(32, 4, activation='relu', padding='same')(up4)
        conv_d3_2 = Conv1D(32, 4, activation='relu', padding='same')(conv_d3)
        up3 = UpSampling1D(2)(conv_d3_2)

        #2 
        conv_d2_2 = Conv1D(64, 4, activation='relu', padding='same')(up3)
        conv_d2 = Conv1D(32, 30, activation='relu', padding='same')(conv_d2_2)
        up2 = UpSampling1D(2)(conv_d2)

        #1
        conv_d1 = Conv1D(32, 4, activation='relu', padding='same')(up2)
        up1 = UpSampling1D(2)(conv_d1)

        #out
        rec_signal = Dense(12, activation='sigmoid')(up1)

        autoencoder = Model(input=input_shape, output=rec_signal)
        autoencoder.compile(optimizer='adam', loss='mse')

        self.current_model = autoencoder

    def get_10x_autoencoder(self):
        input_shape = self.input_shape

        #encoder
        #1
        conv_e1 = Conv1D(32, 20, activation='relu', padding='same')(
            input_shape)
        pool_e1 = MaxPooling1D(2, padding='same')(conv_e1)

        #2
        conv_e2 = Conv1D(64, 4, activation='relu', padding='same')(
            pool_e1)
        conv_e2_2 = Conv1D(64, 4, activation='relu', padding='same')(
            conv_e2)
        pool_e2 = MaxPooling1D(2, padding='same')(conv_e2_2)

        #3
        conv_e3 = Conv1D(32, 4, activation='relu', padding='same')(
            pool_e2)
        conv_e3_2 = Conv1D(32, 4, activation='relu', padding='same')(
            conv_e3)
        pool_e3 = MaxPooling1D(2, padding='same')(conv_e3_2)

        #4
        conv_e4 = Conv1D(32, 4, activation='relu', padding='same')(
            pool_e3)
        conv_e4_2 = Conv1D(16, 4, activation='relu', padding='same')(
            conv_e4)
        pool_e4 = MaxPooling1D(2, padding='same')(conv_e4_2)


        #4
        conv_d4 = Conv1D(32, 4, activation='relu', padding='same')(pool_e4)
        conv_d4_2 = Conv1D(32, 4, activation='relu', padding='same')(conv_d4)
        up4 = UpSampling1D(2)(conv_d4_2)


        #3
        conv_d3 = Conv1D(32, 4, activation='relu', padding='same')(up4)
        conv_d3_2 = Conv1D(32, 4, activation='relu', padding='same')(conv_d3)
        up3 = UpSampling1D(2)(conv_d3_2)

        #2 
        conv_d2_2 = Conv1D(64, 4, activation='relu', padding='same')(up3)
        conv_d2 = Conv1D(32, 30, activation='relu', padding='same')(conv_d2_2)
        up2 = UpSampling1D(2)(conv_d2)

        #1
        conv_d1 = Conv1D(32, 4, activation='relu', padding='same')(up2)
        up1 = UpSampling1D(2)(conv_d1)

        #out
        rec_signal = Dense(12, activation='sigmoid')(up1)

        autoencoder = Model(input=input_shape, output=rec_signal)
        autoencoder.compile(optimizer='adam', loss='mse')

        self.current_model = autoencoder

    def get_3x_autoencoder(self):
        input_shape = self.input_shape

        #encoder
        #1
        conv_e1 = Conv1D(32, 20, activation='relu', padding='same')(
            input_shape)
        pool_e1 = MaxPooling1D(2, padding='same')(conv_e1)

        #2
        conv_e2 = Conv1D(64, 4, activation='relu', padding='same')(
            pool_e1)
        conv_e2_2 = Conv1D(64, 4, activation='relu', padding='same')(
            conv_e2)
        pool_e2 = MaxPooling1D(2, padding='same')(conv_e2_2)

        #3
        conv_e3 = Conv1D(32, 4, activation='relu', padding='same')(
            pool_e2)
        conv_e3_2 = Conv1D(32, 4, activation='relu', padding='same')(
            conv_e3)
        pool_e3 = MaxPooling1D(2, padding='same')(conv_e3_2)


        #3
        conv_d3 = Conv1D(32, 4, activation='relu', padding='same')(pool_e3)
        conv_d3_2 = Conv1D(32, 4, activation='relu', padding='same')(conv_d3)
        up3 = UpSampling1D(2)(conv_d3_2)

        #2 
        conv_d2_2 = Conv1D(64, 4, activation='relu', padding='same')(up3)
        conv_d2 = Conv1D(32, 30, activation='relu', padding='same')(conv_d2_2)
        up2 = UpSampling1D(2)(conv_d2)

        #1
        conv_d1 = Conv1D(32, 4, activation='relu', padding='same')(up2)
        up1 = UpSampling1D(2)(conv_d1)

        #out
        rec_signal = Dense(12, activation='sigmoid')(up1)

        autoencoder = Model(input=input_shape, output=rec_signal)
        autoencoder.compile(optimizer='adam', loss='mse')

        self.current_model = autoencoder

    def get_encoder(self):
        encoded_layer = self.current_model.layers[int(len(self.current_model.layers)/2)]
        self.encoder = Model(self.current_model.inputs, encoded_layer.output)

    def get_error_by_input(self):
        try:
            self.norm_errors = self._return_all_errors(self.normal_testing)
        except:
            print('Could not get normal testing errors')
        try:
            self.afib_errors = self._return_all_errors(self.afib_data)
        except:
            print('Could not get afib testing errors')

    def _return_all_errors(self, input_data):
        test_error = np.zeros(len(input_data[:, 0, 0]))
        for i in range(0, len(test_error)):
            current_input = np.expand_dims(input_data[i,:,:], axis=0)
            test_error[i] = self.current_model.evaluate(current_input, current_input)

        return test_error

    def load_existing_model(self, file_name):
        self.current_model = load_model(file_name)

    def run_autoencoder(self, val_split=.2, n_batch=32, n_epochs=100):
        self.history = self.current_model.fit(self.normal_training,
                                              self.normal_training,
                                              validation_split=val_split,
                                              verbose=1,
                                              batch_size=n_batch,
                                              epochs=n_epochs)

    def plot_history(self):
        if not self.history:
            return

        try:
            import matplotlib.pyplot as plt
            history = self.history

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
        except:
            print('Could not plot loss')

    def evaluate_model(self):
        if not self.history:
            return

        self.norm_eval = self.current_model.evaluate(self.normal_testing,
                                                     self.normal_testing)
        self.afib_eval = self.current_model.evaluate(self.afib_data,
                                                     self.afib_data)

    def predict_test_data(self, is_plotted=False, n_bin=40, n_range=.02):
        if not self.norm_predict:
            self.norm_predict = self.current_model.predict(self.normal_testing)
            self.afib_predict = self.current_model.predict(self.afib_data)

        if is_plotted:
            plt.hist(self.norm_errors,
                     bins=n_bin, alpha=0.5, range=(0, n_range))
            plt.hist(self.afib_errors,
                     bins=n_bin, alpha=0.5, range=(0, n_range))
            plt.show()


    def encode_data(self):
        encoder = self.encoder

        norm_encoded = encoder.predict(self.normal_testing)
        num_obs = len(norm_encoded[:,0,0])
        num_cols = len(norm_encoded[0,:,0])*len(norm_encoded[0,0,:])
        self.norm_encoded_flat = norm_encoded.reshape(
            (num_obs, num_cols), order='f')

        afib_encoded = encoder.predict(self.afib_data)
        num_obs = len(afib_encoded[:,0,0])
        num_cols = len(afib_encoded[0,:,0])*len(norm_encoded[0,0,:])
        self.afib_encoded_flat = afib_encoded.reshape(
            (num_obs, num_cols), order='f')

    def get_pca_encoded(self, is_plotted=False):
        pca_inputs = np.concatenate((self.norm_encoded_flat, 
                                     self.afib_encoded_flat))
        pca = PCA(n_components=12)
        components = pca.fit_transform(pca_inputs)

        if is_plotted:
            import pdb
            #pdb.set_trace()
            num_normal = len(self.norm_encoded_flat[:,0])
            plt.scatter(components[0:num_normal,0], components[0:num_normal,1])
            plt.scatter(components[num_normal:,0], components[num_normal:,1])
            plt.show()

        return components


    def plot_random_test_wave(self, wave_type='normal'):
        rand_num = random.random()
        if wave_type == 'normal':
            rand_num = int(len(self.normal_testing[:,0,0])*rand_num)
            original_wave = self.normal_testing[rand_num,:,:] 
            autoencoded_wave = self.norm_predict[rand_num,:,:]
        elif wave_type == 'afib':
            rand_num = int(len(self.afib_data[:,0,0])*rand_num)
            original_wave = self.afib_data[rand_num,:,:]
            autoencoded_wave = self.afib_predict[rand_num,:,:]
        else:
            return

        self.plot_twelve_lead(pd.DataFrame(original_wave),
                              pd.DataFrame(autoencoded_wave))
        
    def plot_twelve_lead(self, original, prediction):
        original.plot(subplots=True,
                            layout=(6, 2),
                            figsize=(6, 6),
                            sharex=False,
                            sharey=False,
                            legend=False,
                            style=['k' for i in range(12)])
        axes = plt.gcf().get_axes()
        index = 0
        for ax in axes:
            prediction.iloc[:,index].plot(ax=ax, style='b')
            ax.axis('off')
            index+=1
            
        plt.show()


