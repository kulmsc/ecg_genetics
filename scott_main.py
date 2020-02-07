from scott_ml_models import Models
import numpy as np
import os
import pdb
#import matplotlib.pyplot as plt
from keras.models import load_model
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def justWrite(fileName,toWrite):
    with open(fileName,'w') as f:
        for line in toWrite:
            #writeLine = [str(i) for i in line]
            #f.write('\t'.join(writeLine)+'\n')
            f.write(str(line)+'\n')

learning_rates = [10**-4]
l1_nodes = [64, 256]
l2_nodes = [0, 64]
include_stat = True
dropout_second = [True, False]
all_auc = []
all_acc = []


for l1 in l1_nodes:
    for l2 in l2_nodes:
        for ds in dropout_second:
            for lr in learning_rates:

                if l2 == 0 and ds:
                    pass

                else:
                    #Data
                    path_to_afibs = 'data/ecg/afib_pickled_ind'
                    path_to_normals = 'data/ecg/normal_pickled_ind'
                    path_to_genetic = '/home/kulmsc/athena/ecg_kulm/data/genetic/withEcg/afibSnps.raw'
                    path_to_genetic = '/home/kulmsc/athena/ecg_kulm/data/genetic/noEcg/afibSnps.raw'

                    model_obj = Models()
                    num_obs = 2500

                    model_obj.normal_training = []
                    model_obj.load_data(path_to_normals, path_to_genetic, num_obs, rhythm_type='normal')
                    pdb.set_trace()
                    model_obj.load_data(path_to_afibs, path_to_genetic, num_files=300, rhythm_type='afib')
                    model_obj.load_genetic_data(path_to_genetic, True)

                    model_obj.get_genetic_nn(l1, l2, True, ds, lr)
                    model_obj.cv_genetic_nn()

                    print("DONE")
                    print(model_obj.genetic_auc)
                    print(model_obj.genetic_acc)

                    all_auc.append(model_obj.genetic_auc)
                    all_acc.append(model_obj.genetic_acc)

                    print("DOWN")
                    print(all_auc)
                    print(all_acc)

                    justWrite("check_auc", all_auc)
                    justWrite("check_acc", all_acc)

