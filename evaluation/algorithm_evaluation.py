#
# ------------------------------------------------------------------------------------------------------%
# Created by "Ahmed Gad"                                                                                %
#                                                                                                       %
#       Email:      ahmed.gad_it@yahoo.com                                                              %
#       Homepage:   https://www.researchgate.net/profile/Ahmed_G_Gad                                    %
#       Github:     https://github.com/ahmedgad19891                                                    %
#-------------------------------------------------------------------------------------------------------%

from numpy import shape, array, zeros, ones, ceil
from copy import deepcopy
from pandas import DataFrame
from time import time
from os import getcwd, path, makedirs
from evaluation.utils import Utils

class Evaluation(Utils):

        ID_FIT = 0      # best fintness
        ID_ACC = 1      # best accuracy
        ID_FEAT = 2    # best selected features
        ID_TIME = 3     # best processing time
        ID_LOSS = 4     # best loss train

        def __init__(self, dataset_list=None, algo_dicts=None, obj_fun_dicts=None, trans_fun_dicts=None, num_runs=30, domain_range=(0,1), log=True,
                     epoch=100, pop_size=30, lsa_epoch=10, f_combina=None):
            
                Utils.__init__(self, dataset_list, algo_dicts, obj_fun_dicts, trans_fun_dicts)
                self.num_runs = num_runs
                self.domain_range = domain_range
                self.log = log
                self.epoch = epoch
                self.pop_size = pop_size
                self.lsa_epoch = lsa_epoch
                self.f_combina = f_combina
                
        def _eval__(self):
                dataset_splits = self._db_handler_Relief__()
               
                
                if len(self.algo_dicts) == 1:
                        fit_dict, acc_dict, feat_dict, time_dict, loss_dict = ( dict.fromkeys(self.dataset_list , []) for _ in range(len(self.metrics)+1) )
                
                for obj_fun_cat, obj_fun_list in self.obj_fun_dicts.items():
                        
                        for ObjFunc in obj_fun_list:
                                
                                for trans_func_cat, trans_func_list in self.trans_fun_dicts.items():
                                        if len(self.algo_dicts) == 1:
                                                result_path = getcwd() + "/history"
                                                if not path.exists(result_path):
                                                        makedirs(result_path)
                                        else:
                                                self._dir_maker__(ObjFunc, trans_func_cat)
                                        
                                        for trans_func in trans_func_list:
                                                temp_fit_dict, temp_acc_dict, temp_feat_dict, temp_time_dict, temp_loss_dict = ( dict.fromkeys(self.dataset_list , []) for _ in range(len(self.metrics)+1) )
                                                
                                                for ds in self.dataset_list:
                                                        fit_list, acc_list, feat_list, time_list, loss_list = ( [] for _ in range(len(self.metrics)+1) )

                                                        self.problem_size = shape(dataset_splits[ds][0])[1]
                                                        
                                                        
                                                        ml = ObjFunc(*dataset_splits[ds])
                                                        before_accuracy = 1 - ml._fit__(ones(self.problem_size))
                                                        
                                                        
                                                        for name, Algo in self.algo_dicts.items():
                                                               
                                                                     
                                                                param_result_path = getcwd() + "/Final_BNRO/Final_BNRO_result.doc"
                                                                     
                                                                print("> Dataset: {}, Objective Function: {} ==> {}, Transfer Function: {} ==> {}, Original Dataset Shape (Samples ==> {}, Features ==> {}), Original Training Dataset (Samples ==> {}, Features ==> {}), Original Testing Dataset (Samples ==> {}, Features ==> {}), Model Name: {}"
                                                                      .format(ds, obj_fun_cat, ObjFunc.__name__, trans_func_cat, trans_func.__name__, shape(dataset_splits[ds][0])[0] + shape(dataset_splits[ds][1])[0] , shape(dataset_splits[ds][0])[1], shape(dataset_splits[ds][0])[0], shape(dataset_splits[ds][0])[1], shape(dataset_splits[ds][1])[0], shape(dataset_splits[ds][1])[1], name))
                                                                
                                                                print("-------------------------------------------------------------------------")
                                                                print("^^^Accuracy of Original Dataset (Before Feature Selection) {:.4f}".format(before_accuracy))
                                                                
                                                                                                                          
                                                                if  path.exists(param_result_path):
                                                                     param_file=open(param_result_path,'a')
                                                                     param_file.write("-------------------------------------------------------------------------")
                                                                     param_file.write("\nAlgorithm:  {} >>>>\n\n".format(name))
                                                                     param_file.write(">>>> Dataset:  {}\n     Objective Function:  {}  ==>   {}\n     Transfer Function:   {}  ==>  {}\n\n     Original Dataset Shape (Samples ==> {}, Features ==> {})\n     Original Training Dataset (Samples ==> {}, Features ==> {})\n     Original Testing Dataset (Samples ==> {}, Features ==> {})\n\n"
                                                                      .format(ds, obj_fun_cat, ObjFunc.__name__, trans_func_cat, trans_func.__name__,  shape(dataset_splits[ds][0])[0] + shape(dataset_splits[ds][1])[0] , shape(dataset_splits[ds][0])[1], shape(dataset_splits[ds][0])[0], shape(dataset_splits[ds][0])[1], shape(dataset_splits[ds][1])[0], shape(dataset_splits[ds][1])[1]))    
                                                                     param_file.write("-------------------------------------------------------------------------")
                                                                     param_file.write("\n\n***Accuracy of Original Dataset (Before Feature Selection): {:.4f}\n\n".format(before_accuracy))
                                                                     param_file.close()

                                                                res = array([zeros((5), dtype=object) for _ in range(self.num_runs)])



                                                                for id_runs in range(self.num_runs):
                                                                        start_time = int(round(time() * 1000))
                                                                        print("Run: ", id_runs)
                                                                          
                                                                        md = Algo(ml._fit__, trans_func, self.problem_size, self.domain_range, self.log,
                                                                                  self.epoch, self.pop_size, self.lsa_epoch, id_runs)
                                                                                                                                                
                                                                        best_pos, best_fit, loss_train = md._train__()
                                                                        time_required = int(round(time() * 1000)) - start_time
                                                                        after_accuarcy = self._get_accuracy__(best_pos, best_fit)
                                                                        res[id_runs][self.ID_FIT] = best_fit
                                                                        res[id_runs][self.ID_ACC] = after_accuarcy
                                                                        res[id_runs][self.ID_FEAT] = best_pos
                                                                        res[id_runs][self.ID_TIME] = time_required
                                                                        res[id_runs][self.ID_LOSS] = loss_train

                                                                result, best_features, worst_features = self._sort_and_get_metrics(res, self.ID_FIT, self.ID_MIN_PROB, self.ID_MAX_PROB)

                                                                fit_list.extend(result[self.ID_FIT])
                                                                acc_list.extend(result[self.ID_ACC])
                                                                feat_list.extend(result[self.ID_FEAT])
                                                                time_list.extend(result[self.ID_TIME])
                                                                loss_list.extend([result[self.ID_LOSS]])
                          
                                                                  
                                                                print("-------------------------------------------------------------------------")                                                               
                                                                print("^^^After Feature Selection:")
                                                                print("***Best fitness: {:.4f}, Mean fitness: {:.4f}, Worst fitness: {:.4f}, Fitness SD: {:.4f}"
                                                                      .format(*result[self.ID_FIT]))
                                                                print("***Best accuracy: {:.4f}, Mean accuracy: {:.4f}, Worst accuracy: {:.4f}, Accuracy SD: {:.4f}"
                                                                      .format(*result[self.ID_ACC]))
                                                                print("***Best feature size: {0}/{5} ==> {6}, Mean feature size: {1:.4f}, Feature selection ratio: {4:.4f}, Worst feature size: {2}/{5} ==> {7}, Feature size SD: {3:.4f}"
                                                                      .format(*result[self.ID_FEAT], self.problem_size, best_features, worst_features))
                                                                print("***Best processing time: {:.4f}, Mean processing time: {:.4f}, Worst processing time: {:.4f}, Processing time SD: {:.4f}"
                                                                      .format(*result[self.ID_TIME]))                                                                
                                                                print("-------------------------------------------------------------------------")
                                                                print("-------------------------------------------------------------------------")

                                                                
                                                                if  path.exists(param_result_path):
                                                                    param_file=open(param_result_path,'a')
                                                                                                                                            
                                                                    param_file.write("-------------------------------------------------------------------------")                                                                                                             
                                                                    param_file.write("\n\n^^^After Feature Selection:")                                                                    
                                                                    param_file.write("\n***Best fitness: {:.4f}, Mean fitness: {:.4f}\nWorst fitness: {:.4f}, Fitness STD: {:.4f}"
                                                                          .format(*result[self.ID_FIT]))
                                                                    param_file.write("\n\n***Best accuracy: {:.4f}, Mean accuracy: {:.4f}\nWorst accuracy: {:.4f}, Accuracy STD: {:.4f}"
                                                                          .format(*result[self.ID_ACC]))
                                                                    param_file.write("\n\n***Best feature size: {0}/{5} ==> {6}, Mean feature size: {1:.4f}, Feature selection ratio: {4:.4f}\nWorst feature size: {2}/{5} ==> {7}, Feature size STD: {3:.4f}"
                                                                          .format(*result[self.ID_FEAT], self.problem_size, best_features, worst_features))
                                                                    param_file.write("\n\n***Best processing time: {:.4f}, Mean processing time: {:.4f}\nWorst processing time: {:.4f}, Processing time STD: {:.4f}\n\n"
                                                                          .format(*result[self.ID_TIME]))                                                                
                                                                    param_file.write("-------------------------------------------------------------------------")                                         
                                                                    param_file.write("-------------------------------------------------------------------------")                                         
                                                                    
                                                                    param_file.close()
                                                                    
                                                                    
                                                        print("===================================================================================")

                                                        temp_fit_dict[ds] = fit_list
                                                        temp_acc_dict[ds] = acc_list
                                                        temp_feat_dict[ds] = feat_list
                                                        temp_time_dict[ds] = time_list
                                                        temp_loss_dict[ds] = loss_list

                                                        if len(self.algo_dicts) == 1:
                                                                fit_dict[ds] = deepcopy(fit_dict[ds] + fit_list)
                                                                acc_dict[ds] = deepcopy(acc_dict[ds] + acc_list)
                                                                feat_dict[ds] = deepcopy(feat_dict[ds] + feat_list)
                                                                time_dict[ds] = deepcopy(time_dict[ds] + time_list)
                                                                loss_dict[ds] = deepcopy(loss_dict[ds] + loss_list)

                                                print("===================================================================================\n"*2)

                                                if len(self.algo_dicts) > 1:
                                                        self._output_results__(dfs=[temp_fit_dict, temp_acc_dict, temp_feat_dict, temp_time_dict],
                                                                               path_list=self.path_list,
                                                                               head=[self.algo_dicts.keys()],
                                                                               out_path=[ObjFunc.__name__, trans_func.__name__])
                                                        self._output_pkls_and_figs__(temp_loss_dict, self.epoch, ObjFunc.__name__, trans_func.__name__)

                if len(self.algo_dicts) == 1:
                        self._output_results__(dfs=[fit_dict, acc_dict, feat_dict, time_dict],
                                               path_list=[getcwd() + "/history"]*4,
                                               head=[self.algo_dicts.keys(), self.obj_funs, self.trans_funcs],
                                               out_path=[obj_fun_cat, "algorithms_overall"])

