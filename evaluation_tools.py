import torch
import pandas as pd
import numpy as np
from probes import LRProbe, TTPD, LRProbePCA
from sklearn.metrics import roc_auc_score
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import os
import pickle

def prepare_data(data_root, ds_names, train_set_sizes, same_size=True):

  """
  args:
      data_root: path to a format dataset
      ds_names: list of topics to be selected from that dataset
      train_set_sizes: sizes to be sampled per topic (same order as ds_names)
      same_size: balance all topics to be equal size?
  returns: 
      torch tensors with all the activations of all the topics from that format combined
  """

  acts=[]
  labels=[]
  polarities=[]
  acts_centered=[]

  for ds_name in ds_names:

    data_dict=torch.load(f"{data_root}/{ds_name}.pt",map_location=torch.device('cpu'))

    act=data_dict["acts"]
    label=data_dict["labels"]

    if same_size is True:

      rand_subset=np.random.choice(len(act), min(train_set_sizes), replace=False)
      act=act[rand_subset]
      label=label[rand_subset]

    polarity=torch.full((label.shape[0],), float(1))
    if "neg_" in ds_name:
      polarity=torch.full((label.shape[0],), float(-1))
 
    acts.append(act)
    labels.append(label)
    polarities.append(polarity)
    acts_centered.append(act-torch.mean(act,dim=0))

  acts=torch.cat(acts)
  labels=torch.cat(labels)
  polarities=torch.cat(polarities)
  acts_centered=torch.cat(acts_centered)

  return acts, labels, polarities, acts_centered

def dataset_sizes(data_root, ds_names):
  sizes=[]
  for ds_name in ds_names:
    data_dict=torch.load(f"{data_root}/{ds_name}.pt",map_location=torch.device('cpu'))
    sizes.append(len(data_dict["acts"]))

  return sizes


def compute_metrics_crossval(root_path_train,  root_path_test, topic_names, probes, N=20):


  """
  computes the average generalization accuracy between two format types while also performing cross validation over various topics.

  args:
    -root_path_train: path to the format that is to be used as a train set
    -root_path_test: path to the format that is to be used as the test set
    -topic_names: names of the different topic names for crossvalidation [name, neg_name, name2, neg_name2], where each   dataset has to be followed by its version with negations
    -probes: names of the probes for which the accuracy is to be computed (see dict for options)
    -N number of times to repeat the experiments to remove randomness from rebalancing the training size datasets
  """



  accuracies_dict={}
  tupled_names_train=[(topic_names[i], topic_names[i + 1]) for i in range(0, len(topic_names), 2)]

  #we loop over all tuples of "ds_name+neg_ds_name" and remove one pair each time as a crossvalidation
  for count, (ds_name, neg_ds_name) in enumerate(tupled_names_train):

    accuracies_dict[ds_name]={ "LRC": np.nan , "LR": np.nan, "TTPD": np.nan, "PCA": np.nan}
    test_datasets=[ds_name, neg_ds_name]
    train_datasets=[x for x in topic_names if x not in test_datasets]
    
    # train_datasets=[x for x in train_ds_names if x not in [ds_name, neg_ds_name]]
    train_set_sizes=dataset_sizes(root_path_train, train_datasets)
    probe_dict={"LRC": LRProbe, "LR": LRProbe, "TTPD": TTPD, "PCA": LRProbePCA}

    for probe_name in probes:

      probe_class=probe_dict[probe_name]
      accuracies=[]

      for i in range(N):
        
        train_acts, train_labels, train_polarities, train_acts_centered =prepare_data(root_path_train, train_datasets,train_set_sizes, same_size=True)
        test_acts, test_labels, test_polarities, test_acts_centered=prepare_data(root_path_test,  test_datasets ,train_set_sizes , same_size=False)
 
        if probe_name=="TTPD":
          probe=probe_class.from_data(train_acts_centered, train_acts, train_labels, train_polarities)
          pred=probe.pred(test_acts)

        elif (probe_name=="LRC") or (probe_name=="PCA"):
          probe=probe_class.from_data(train_acts_centered, train_labels)
          pred=probe.pred(test_acts_centered)

        elif probe_name=="LR":
          probe=probe_class.from_data(train_acts, train_labels)
          pred=probe.pred(test_acts)          


        accuracy=(pred==test_labels).sum()/len(test_labels)
        accuracies.append(accuracy)
 
      avg_accuracy=np.mean(accuracies)
      accuracies_dict[ds_name][probe_name]=avg_accuracy

  average_accuracy_dict={}

  for probe_name in probe_dict.keys():

    accuracies=[]

    for key in accuracies_dict.keys():
      accuracies.append(accuracies_dict[key][probe_name])

    average_accuracy_dict[probe_name]=np.mean(accuracies)

  return accuracies_dict, average_accuracy_dict

 

 