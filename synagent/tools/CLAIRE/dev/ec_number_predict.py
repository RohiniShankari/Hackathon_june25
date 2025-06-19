import sys,os
sys.path.append('/home/boltzmann-labs/synagent/tools/CLAIRE')
from dev.prediction.inference_EC import inference
import pickle
import numpy as np
import pandas as pd
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
from typing import List
from drfp import DrfpEncoder
import inspect,os
path = inspect.getfile(inspect.currentframe())
# print(path)
dir_path = os.path.dirname(os.path.abspath(path))
# import sys
# sys.path.append(dir_path)
# path_temp=dir_path.split('/')
# idx=path_temp.index('REBOLT-FEATURES')
# DIR_PATH='/'.join(path_temp[:idx+1])
def get_ec_numbers_from_rxn(rxns: List[str]) -> list:
    """
    Get EC numbers from a reaction string.
    
    Args:
        rxns (str): The list of reaction string in SMILES format in the form [reactants>>product,...].
        
    Returns:
        list: A list of EC numbers associated with the reaction.
    """
    # Placeholder for actual logic to extract EC numbers
    # os.system('python3 /home/boltzmann-labs/synagent/tools/CLAIRE/dev/test.py') 
    fps=DrfpEncoder.encode(rxns,n_folded_length=256)
    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    # example_rxns = ["NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.NCCC=O.O>>NCCC(=O)O", "C=C(C)CCOP(=O)([O-])OP(=O)([O-])[O-].CC(C)=CCOP(=O)(O)OP(=O)(O)O>>CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP(=O)(O)OP(=O)(O)O", "N.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=C([O-])CCC(=O)C(=O)[O-].[H+]>>N[C@@H](CCC(=O)[O-])C(=O)[O-]"]
    rxnfp = rxnfp_generator.convert_batch(rxns)
    pickle.dump(rxnfp, open(dir_path+'/rxnfp_emb.pkl', 'wb'))
    # drfp = pickle.load(open('dev/my_rxn_fps.pkl', 'rb'))
    drfp=fps
    rxnfp = pickle.load(open(dir_path+'/rxnfp_emb.pkl', 'rb'))
    test_data = []

    for ind, item in enumerate(rxnfp):
        rxn_emb = np.concatenate((np.reshape(item, (1,256)), np.reshape(drfp[ind], (1,256))), axis=1)
        test_data.append(rxn_emb)

    test_data = np.concatenate(test_data,axis=0)

    train_data = pickle.load(open (dir_path+'/data/model_lookup_train.pkl', 'rb'))
    train_labels = pickle.load(open (dir_path+'/data/pred_rxn_EC123/labels_train_ec3.pkl', 'rb')) #if you want 1-level EC or 2-level EC, change it to pred_rxn_EC1/labels_trained_ec1.pkl or pred_rxn_EC12/labels_trained_ec2.pkl, resepetively.
    # input your test_labels
    test_labels = None
    # print("test_labels is None, so we will not evaluate the model")
    print(rxns)    
    test_tags = [rxns[i] for i in range(len(rxns))]

    # EC calling results using maximum separation
    pretrained_model = dir_path+'/results/model/pred_rxn_EC123/layer5_node1280_triplet2000_final.pth'
    inference(train_data, test_data, train_labels, test_tags,test_labels, pretrained_model, evaluation=False, topk=3, gmm = dir_path+'/gmm/gmm_ensumble.pkl')
    df=pd.read_csv(dir_path+'/results/test_new_prediction.csv',header=None)
    return list(df.values)
