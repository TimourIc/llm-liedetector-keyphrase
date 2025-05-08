import os
import urllib.request
import pandas as pd
import copy
import numpy as np
import pickle
import sympy
from format_templates import return_types, type1format, type2format, type3format

###SOME USEFUL VARS
topics=["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "inventors", "neg_inventors", "animal_class", "neg_animal_class", "element_symb", "neg_element_symb", "facts", "neg_facts"]
raw_data="data/raw"
proc_data="data/proc"
columns = ["prompt", "statement", "statement_label", "bool", "label"]


###DEFINE THREE TYPES OF FORMATTING FUNCTIONS
def main(preformat_dict, extra_name, seed_count=1):

    """
    Given a dictionary of strings that represent the three base formats with some extra part added to them, creates the conversation datasets.
        args:
           preformat_dict: dictionary for the three prompt types (with some extra part added) with replaceable strings to still be filled in
           extra_name: for saving purposes, the name of the extra_part that was added to the three formats above
           seed_count_start: parameter that allows to control the random seeding
        returns:
            * no output, but saves a dataset of conversations for those three formats.
    """


    #make some placeholder dictionaries the formats to be saved
    original={ds_name: pd.DataFrame(columns=columns) for ds_name in topics}
    format1={ds_name: pd.DataFrame(columns=columns) for ds_name in topics}
    format2={ds_name: pd.DataFrame(columns=columns) for ds_name in topics}
    format3={ds_name: pd.DataFrame(columns=columns) for ds_name in topics}

    #read in all the original data statement dataset
    dfs={ds_name: pd.read_csv(f"{raw_data}/{ds_name}.csv") for ds_name in topics}

    for count, ds_name in enumerate(topics):

        statements=dfs[ds_name].statement
        original_labels=dfs[ds_name].label

        #original
        df=original[ds_name]
        type1format(df, statements, original_labels, "original", preformat_dict)

        #format1
        df=format1[ds_name]
        type1format(df, statements, original_labels, "format1", preformat_dict)
 
        #format2
        df=format2[ds_name]

        "We create the seed numbers in such a way that they are always unique across the various topics, but identical accross formats"
        "Note that this is fine for our purposes because we will always be cross-validating across topics, so the training set will never have the same seed randomness as the test set."

        unique_numer1=sympy.prime(1)*sympy.prime(4+count)
        np.random.seed(unique_numer1)
        bools = pd.Series(np.random.randint(0, 2, size=len(statements)))
        type2format(df, statements, original_labels,bools, "format2", preformat_dict)

        #format3
        df=format3[ds_name]
        unique_numer2=sympy.prime(3)*sympy.prime(6+count)
        np.random.seed(unique_numer2)
        bools = pd.Series(np.random.randint(0, 2, size=len(statements)))
        type3format(df, statements,  statements.sample(frac=1, random_state=unique_numer1, ignore_index=True),statements.sample(frac=1, random_state=unique_numer2, ignore_index=True), original_labels, bools, "format3",preformat_dict)

        formats=[format1, format2, format3]


        #Save all formats 
        os.makedirs("data/proc", exist_ok=True)
        with open( f"{proc_data}/original.pkl","wb") as f:
            pickle.dump(original, f)
            

        os.makedirs("data/proc", exist_ok=True)
        for i in range(0,3):
            with open( f"{proc_data}/format{i+1}_{extra_name}.pkl","wb") as f:
                pickle.dump(formats[i], f)
        
        seed_count+=1

    return seed_count
    

if __name__=="__main__":

    
    extra_names={"none", "longer", "keyphrase", "longer_keyphrase", "control", "longer_control"}

    for extra_name in extra_names:
        
        preformat_dict= return_types(extra_name)
        main(preformat_dict, extra_name)
     

 





       






