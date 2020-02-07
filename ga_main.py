from ga_ml_models import Models
from datetime import datetime
import numpy as np
import gc
import random
import os
import pdb
import pickle
#import matplotlib.pyplot as plt
from keras.models import load_model
os.environ['KMP_DUPLICATE_LIB_OK']='True'







class genetic():
    def __init__(self, n_pop, keep_frac, reject_frac):
        self.n_pop = n_pop
        self.keep_frac = keep_frac
        self.save_num = round(n_pop * keep_frac)
        self.reject_keep = round(n_pop * reject_frac)

        self.path_to_afibs = 'data/ecg/afib_pickled_ind' #for ecg data
        self.path_to_normals = 'data/ecg/normal_pickled_ind' #for ecg data
        self.path_to_genetic = '/home/kulmsc/athena/ecg_kulm/data/genetic/noEcg/afibSnps.2276.1e-4.1e-4.0.5.raw'
        self.path_to_pheno = '/home/kulmsc/athena/ecg_kulm/data/genetic/exclude.250.pheno'

        time_list = list(datetime.now().timetuple())[0:5]
        time_name = '-'.join([str(i) for i in time_list])
        time_name = "genealgo."+time_name+".res"
        self.add_line(time_name, ["lr","l1","l2","l3","d1","d2","d3","act","optim"], ["test_acc", "valid_acc", "valid_auc", "loss_change", "acc_change"])
        self.time_name = time_name

        include_stat = True
        learning_rates = [10**-4, 10**-6, 10**-8]
        l1_nodes = [4, 24, 64, 256]
        l2_nodes = [0, 2, 16]
        l3_nodes = [0, 2, 8]
        dropout_first = [0.2, 0.5, 0.8]
        dropout_second = [True, False]
        dropout_third = [True, False]
        activator = ["elu", "softmax", "relu", "sigmoid"]
        optimizer = ["adam", "adamax", "rmsprop"]
        self.all_possible_opts = {"lr":learning_rates, "l1":l1_nodes, "l2":l2_nodes, "l3":l3_nodes, "d1":dropout_first, 
                                  "d2":dropout_second, "d3":dropout_third, "act":activator, "optim":optimizer}

        self.data_obj = self.get_genome("data")
        self.all_nets = list()
        self.used_params = list()
        if os.path.isfile("pickup_params.pl"):
            start_params = pickle.load( open( "pickup_params.pl", "rb" ) )

        for i in range(n_pop):
            if os.path.isfile("pickup_params.pl"):
                self.all_nets.append(self.get_genome("nn", start_params[i]))
            else:
                self.all_nets.append(self.get_genome("nn"))
            self.used_params.append(''.join([str(j) for j in self.all_nets[i].opts.values()]))


    def add_line(self, fileName, nn_params, nn_perf):
        toWrite = nn_params + nn_perf
        toWrite = [str(r) for r in toWrite]
        with open(fileName,'a+') as f:
            f.write(str('\t'.join(toWrite))+'\n')



    def get_genome(self, goal, preset = "None"): 

        model_obj = Models()

        if goal == "data":
            model_obj.load_genetic_data(self.path_to_genetic, self.path_to_pheno)

        elif goal == "nn":
            model_obj.normal_testing_genetic = self.data_obj.normal_testing_genetic
            model_obj.normal_training_genetic = self.data_obj.normal_training_genetic
            model_obj.afib_data_genetic = self.data_obj.afib_data_genetic
            if preset == "None":
                model_obj.opts = []
                for k in self.all_possible_opts.keys():
                    model_obj.opts.append(random.choice(self.all_possible_opts[k]))
            else:
                model_obj.opts = preset

            if type(model_obj.opts) != type({'a':1}):
                model_obj.opts = dict(zip(self.all_possible_opts.keys(), model_obj.opts))

            model_obj.get_genetic_nn()
        
        return(model_obj)


    def breed(self):
        """ Produces new neural networks from neural networks that have already been determined to be good """

        print("BREEDING")
        replace_inds = [i for i in range(self.n_pop) if i not in self.keep_net_ind]
        self.poss_genome_breaks = [i for i in range(len(self.all_possible_opts))]
        to_save_param = []

        for ri in replace_inds:
            print(ri)
            #pdb.set_trace()
            counter = 1
            bred_param = self.single_breed(counter)
            if random.uniform(0, 1) < 0.3:
                mutated_param = self.mutate(bred_param)
            else:
                mutated_param = bred_param
            self.all_nets[ri] = self.get_genome("nn", mutated_param)
            self.used_params.append(''.join([str(j) for j in mutated_param]))
        gc.collect()
        return(None)

    def single_breed(self, counter):
        if counter > 10:
            #pdb.set_trace()
            print("too high!")
        print("single breed")
        print(self.keep_net_ind)
        dad_index = random.choice(self.keep_net_ind)
        mom_index = random.choice(self.keep_net_ind[self.keep_net_ind != dad_index])
        xover_spot = random.choice(self.poss_genome_breaks)
        remain_spot = len(self.poss_genome_breaks) - xover_spot
        params = ['x' for i in range(len(self.poss_genome_breaks))]

        for k in range(xover_spot):
            replace_spot = random.choice([j for j in range(len(params)) if params[j] == 'x'])
            replace_param = list(self.all_possible_opts.keys())[replace_spot]
            params[replace_spot] = self.all_nets[mom_index].opts[replace_param]

        for k in range(remain_spot):
            replace_spot = random.choice([j for j in range(len(params)) if params[j] == 'x'])
            replace_param = list(self.all_possible_opts.keys())[replace_spot]
            params[replace_spot] = self.all_nets[dad_index].opts[replace_param]

        if random.uniform(0, 1) < 0.3:
            params = self.mutate(params)


        if ''.join([str(j) for j in params]) in self.used_params:
            print("going again")
            return(self.single_breed(counter + 1))
        else:
            combo_param = ''.join([str(j) for j in params]) 
            self.used_params.append(combo_param)
            return(params)


    def mutate(self, in_param):
        """ Changes one of the parameters within the compiled neural networks """
        keep_param = list(in_param)
        mut_gene = random.choice(list(self.all_possible_opts.keys()))
        #mut_index = [i for i,l in enumerate(self.all_possible_opts.keys()) if l == mut_gene]
        mut_index = list(self.all_possible_opts.keys()).index(mut_gene)
        in_param[mut_index] = random.choice(self.all_possible_opts[mut_gene])
        if ''.join([str(k) for k in in_param]) in self.used_params:
            return(self.mutate(keep_param))
        else:
            return(in_param)


    def evaluate(self):
        """ Gets the AUC of the compiled neural networks """

        print("EVALUATING")
        goReal = True
        all_auc = []
      
        for i in range(len(self.all_nets)):
            if goReal:
                #pdb.set_trace()
                ans = self.all_nets[i].cv_genetic_nn()
                #pdb.set_trace()
                all_auc.append(self.all_nets[i].genetic_auc)
                test_acc = round(ans.history['val_binary_accuracy'][-1], 4)
                valid_auc = round(self.all_nets[i].genetic_auc, 4)
                valid_acc = round(self.all_nets[i].genetic_acc, 4)
                loss_change = round((ans.history['val_loss'][2] - ans.history['val_loss'][-1])*10e6, 4)
                acc_change = round(ans.history['val_binary_accuracy'][2] - ans.history['val_binary_accuracy'][-1], 4)
                param_towrite = [j for j in self.all_nets[i].opts.values()]
                # ["test_acc", "valid_acc", "valid_auc", "loss_change", "acc_change"]
                stat_towrite = [test_acc, valid_acc, valid_auc, loss_change, acc_change]
                self.add_line(self.time_name, param_towrite, stat_towrite)
                #pdb.set_trace()
            else:
                all_auc.append(random.uniform(0.5, 1))

        #pdb.set_trace()
        self.keep_net_ind = np.where(all_auc > np.sort(all_auc)[::-1][self.save_num])[0]

        reject_append = []
        for i in range(self.reject_keep):
            to_add = random.choice([i for i in range(len(all_auc)) if i not in self.keep_net_ind])
            self.keep_net_ind = np.insert(self.keep_net_ind, 0, to_add)
        self.keep_net_ind = np.sort(self.keep_net_ind)
        	

    def save_data(self):
        print("IN SAVE")
        all_opts = []
        for nn in self.all_nets:
            all_opts.append(nn.opts)
        print(all_opts)
        pickle.dump(all_opts, open("pickup_params.pl",'wb'))



#n_gens = 10
#test = genetic(10, 0.2, 0.1)
#for i in range(n_gens):
#    print("gen", i)
#    test.evaluate()
#    test.breed()
#    test.save_data()
#    gc.collect()
#    print("bottom")
#pickle.dump(test, open("finalTest.pl",'wb'))

print("BEGIN")
test = genetic(20, 0.2, 0.1)
print("made test")
test.evaluate()
print("done evaluate")
test.breed()
print("done breed")
test.save_data()
print("saved data")
