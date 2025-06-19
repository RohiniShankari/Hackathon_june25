#!/usr/bin/env python
# -*- coding: utf-8 -*-
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
import joblib
# from rxnfp.transformer_fingerprints import (
#     RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
# )
from rdkit import Chem
# from mol_props_fn import mol_props_fn
import time
import inspect,os
path = inspect.getfile(inspect.currentframe())
# print(path)
dir_path = os.path.dirname(os.path.abspath(path))
# import sys
# sys.path.append(dir_path)
# path_temp=dir_path.split('/')
# idx=path_temp.index('joel')
# DIR_PATH='/'.join(path_temp[:idx+1])

def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, logger=logger, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    features_shards = []
    features_names = []
    for feat_name, feat_path in opt.src_feats.items():
        features_shards.append(split_corpus(feat_path, opt.shard_size))
        features_names.append(feat_name)
    shard_pairs = zip(src_shards, tgt_shards, *features_shards)

    for i, (src_shard, tgt_shard, *features_shard) in enumerate(shard_pairs):
        features_shard_ = defaultdict(list)
        for j, x in enumerate(features_shard):
            features_shard_[features_names[j]] = x
        logger.info("Translating shard %d." % i)
        print(src_shard)
        t=translator.translate(
            src=src_shard,
            src_feats=features_shard_,
            tgt=tgt_shard,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug
            )
    return t[1]
def fwd_feas_prob(rxns,ff_args=None):
    if ff_args:
        rxnfp_generator,ml_model=ff_args
        print('hello')
    else:
        model, tokenizer_bert = get_default_model_and_tokenizer()
        rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer_bert)
        rxn_classifer = joblib.load('/home/ubuntu/joel/retrosynthesis/reactant_knn_cn.sav')
        ml_model=joblib.load('/home/ubuntu/joel/retrosynthesis/ml_extratree.pkl')
    fps = [rxnfp_generator.convert(_) for _ in rxns]
    try:
        probs=ml_model.predict_proba(fps)
    except:
        print(rxns)
        return []
    return probs
    
def translate_smile(opt,tokenized_smile,translator):
    # ArgumentParser.validate_translate_opts(opt)
    # logger = init_logger(opt.log_file)

    # translator = build_translator(opt, logger=logger, report_score=True)
    # src_shards = split_corpus(opt.src, opt.shard_size)
    # tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    # features_shards = []
    # features_names = []
    # for feat_name, feat_path in opt.src_feats.items():
    #     features_shards.append(split_corpus(feat_path, opt.shard_size))
    #     features_names.append(feat_name)
    # shard_pairs = zip(src_shards, tgt_shards, *features_shards)

    # for i, (src_shard, tgt_shard, *features_shard) in enumerate(shard_pairs):
    #     features_shard_ = defaultdict(list)
    #     for j, x in enumerate(features_shard):
    #         features_shard_[features_names[j]] = x
    #     logger.info("Translating shard %d." % i)
    t = translator.translate(
        src=tokenized_smile,
        src_feats={},
        tgt=None,
        batch_size=opt.batch_size,
        batch_type=opt.batch_type,
        attn_debug=opt.attn_debug,
        align_debug=opt.align_debug
        )
    return t[0][0],t[1][0]

def _get_parser():
    parser = ArgumentParser(description='translate.py')
    # parser.add_argument("--config",default="/home/ubuntu/joel/retrosynthesis/train-from-scratch/RtoP/RtoP-MIT-mixed-aug5-translate.yml")
    # parser.defaults(config='/home/ubuntu/joel/retrosynthesis/train-from-scratch/RtoP/RtoP-MIT-mixed-aug5-translate.yml')
    # parser.add_argument("--config", default="/home/ubuntu/joel/retrosynthesis/train-from-scratch/RtoP/RtoP-MIT-mixed-aug5-translate.yml", help="Input file path (default: default_input.txt)")
    # parser.add_argument("--host",default='')
    # parser.add_argument("--port",default=8080)
    # parser.add_argument("api:app",default=8080)
    # parser.add_argument("--reload",default='true')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser
def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    smi=''.join(smi.split(' '))
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    try:
        assert smi == ''.join(tokens)
    except:
        print(smi,''.join(tokens))
    return ' '.join(tokens)
def get_products(smile,translator,opt):
    # parser = _get_parser()

    # opt = parser.parse_args()
    scores,raw_pred=translate_smile(opt,[smi_tokenizer(smile)],translator)
    # print(raw_pred)
    preds=["".join(smile.split(' ')) for smile in raw_pred]
    from rdkit import Chem
    valid_preds=[]
    valid_scores=[]
    for idx,pred in enumerate(preds):
        # pred=pred.replace('<unk>','@')
        try:
            valid_preds.append(Chem.CanonSmiles(pred))
            valid_scores.append(scores[idx].item())
        except:
            continue
    # print(valid_preds)
    return (valid_preds,valid_scores)

def min_max_normalization(data):
    """
    Perform min-max normalization on a list of numeric values.

    Parameters:
    - data: List of numeric values to be normalized.

    Returns:
    - normalized_data: List of normalized values.
    """
    min_val = min(data)
    max_val = 0
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data
def remove_dups(preds,pred_probs):
    d={}
    for idx,pred in enumerate(preds):
        if pred not in d.keys():
            d[pred]=pred_probs[idx]
    return list(d.keys()),list(d.values())
            
def inference(smile,ff_args=None):
    # import argparse
    # parser1=argparse.ArgumentParser()
    parser = _get_parser()
    
    # parser.defaults(parser1)
    # parser.config='/home/ubuntu/joel/retrosynthesis/train-from-scratch/RtoP/RtoP-MIT-mixed-aug5-translate.yml'
    opt = parser.parse_args()
    opt.src=dir_path+'/test_rtop.txt'
    opt.models[0]=dir_path+'/retro_models/USPTO_full_PtoR.pt'
    
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)
    translator = build_translator(opt, logger=logger, report_score=True)
    preds,pred_probs=get_products(smile,translator,opt)
    preds,pred_probs=remove_dups(preds,pred_probs)
    pred_rxns=[smile+'>>'+pred for pred in preds]
    if pred_rxns!=[]:
    #     pred_probs_ml= fwd_feas_prob(pred_rxns,ff_args)
    #     print(pred_probs_ml)
        norm_probs=min_max_normalization(pred_probs)
    #     fwd_feas_probs=[prob[1] for prob in pred_probs_ml]
    # else:
    #     fwd_feas_probs=[]
    #     norm_probs=[]
    res={'reactants':smile,'predictions':preds,"prediction_scores":norm_probs}
    return res
def infernece_single_step(smile,model_path):
    mdl_load_time_start=time.time()
    parser = _get_parser()
    # parser.defaults(parser1)
    # parser.config='/home/ubuntu/joel/retrosynthesis/train-from-scratch/RtoP/RtoP-MIT-mixed-aug5-translate.yml'
    opt = parser.parse_args()
    opt.beam_size=30
    opt.n_best=30
    opt.src='/home/ubuntu/joel/retrosynthesis/test_ptor.txt'
    if model_path =='':
        opt.models[0]=dir_path+'/retro_models/USPTO_full_PtoR.pt'
    else:
        opt.models[0]=model_path
    # print(opt.model)
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)
    translator = build_translator(opt, logger=logger, report_score=True)
    mdl_load_time_end=time.time()
    inf_start=time.time()
    preds,pred_probs=get_products(smile,translator,opt)
    predictions=[]
    scores=[]
    for idx in range(len(preds)):
        if preds[idx] not in predictions:
            predictions.append(preds[idx])
            scores.append(pred_probs[idx])
    inf_end=time.time()
    print('model load time',mdl_load_time_end-mdl_load_time_start)
    print('infernce time',inf_end-inf_start)
    return {"reactants":predictions,"scores":scores}
def inference_boltzmann(smile):
    # import argparse
    # parser1=argparse.ArgumentParser()
    parser = _get_parser()
    # parser.defaults(parser1)
    # parser.config='/home/ubuntu/joel/retrosynthesis/train-from-scratch/RtoP/RtoP-MIT-mixed-aug5-translate.yml'
    opt = parser.parse_args()
    opt.src='/home/ubuntu/joel/retrosynthesis/test_rtop.txt'
    opt.model='/home/ubuntu/joel/retrosynthesis/models/rtop/USPTO-MIT_RtoP_separated.pt'
    
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)
    translator = build_translator(opt, logger=logger, report_score=True)
    preds,pred_probs=get_products(smile,translator,opt)
    preds,pred_probs=remove_dups(preds,pred_probs)
    pred_rxns=[smile+'>>'+pred for pred in preds]
    print(preds)
    print(pred_rxns)
    if pred_rxns!=[]:
        pred_probs_ml= fwd_feas_prob(pred_rxns)
        print(pred_probs_ml)
        norm_probs=min_max_normalization(pred_probs)
        fwd_feas_probs=[prob[1] for prob in pred_probs_ml]
    else:
        fwd_feas_probs=[]
    mol_props=mol_props_fn(preds,norm_probs,fwd_feas_probs)
    out_preds=mol_props.to_dict(orient='records')
    print(out_preds)
    res={'reactants':smile,'predictions':out_preds}
    return res
def main():
    print(inference('CC(=O)C(=O)OCc1ccccc1'))
    # parser = _get_parser()

    # opt = parser.parse_args()
    # # print(opt)
    # opt.models=['/home/ubuntu/joel/retrosynthesis/models/finetuned_model.reactants-products_step_80000.pt']
    
    # # smile='CC(C)(C)N.Cc1ccc(C2=NNC(=O)CC2)cc1OCC1CO1'
    # # scores,raw_pred=translate_smile(opt,[smi_tokenizer(smile)])
    # # preds=["".join(smile.split(' ')) for smile in raw_pred]
    # # from rdkit import Chem
    # # valid_preds=[]
    # # valid_scores=[]
    # # for idx,pred in enumerate(preds):
    # #     try:
    # #         valid_preds.append(Chem.CanonSmiles(pred))
    # #         valid_scores.append(scores[idx].item())
    # #     except:
    # #         continue
            
    # # # s=translate(opt)
    # print(opt)
    # import pandas as pd
    # # file_paths=['/home/ubuntu/joel/manual_reactions/ADDITION_REACTION.csv',
    # #             '/home/ubuntu/joel/manual_reactions/CONDENSATION_REACTION.csv',
    # #             '/home/ubuntu/joel/manual_reactions/CROSS COUPLING_REACTION.csv',
    # #             '/home/ubuntu/joel/manual_reactions/ELIMINATION_REACTION.csv',
    # #             '/home/ubuntu/joel/manual_reactions/HYDROLYSIS_REACTION.csv',
    # #             '/home/ubuntu/joel/manual_reactions/ISOMERIZATION_REACTION.csv',
    # #             '']
    # import os
    # path='/home/ubuntu/joel/manual_reactions/'
    # csv_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # csv_files=[file for file in csv_files if file[-4:]=='.csv']
    # for file_name in tqdm(csv_files):
    #     fail=[]
    #     df=pd.read_csv(os.path.join(path, file_name))
    #     # for i in range(df.shape[0]):
    #     opt.models=[f'/home/ubuntu/joel/retrosynthesis/models/rtop/USPTO-MIT_RtoP_mixed.pt']
    #     # df=df_full[df_full['classes']==i]
    #     print(os.path.join(path, file_name))
    #     print(df['SMILES ID'].to_list())
    #     rcts_raw=[]
    #     prods_raw=[]
    #     unproc_smiles=[]
    #     for rxns_smile in df['SMILES ID'].to_list():
    #         try:
    #             rcts_raw.append(rxns_smile.split('>')[0].strip())
    #             prods_raw.append(rxns_smile.split('>')[-1].strip())
    #             unproc_smiles.append(rxns_smile)
    #         except:
    #             print(rxns_smile)
    #             continue
    #     assert len(prods_raw)==len(rcts_raw)
    #     # rcts_raw=[rxns_smile.split('>')[0].strip() for rxns_smile in df['SMILES ID'].to_list()]
    #     # prods_raw=[rxns_smile.split('>')[-1].strip() for rxns_smile in df['SMILES ID'].to_list()]
    #     rcts=[]
    #     prods=[]
    #     unproc_smile=[]
    #     for idx,r in enumerate(rcts_raw):
    #         rs=r.split('.')
    #         temp=[]
    #         for _ in rs:
    #             temp.append(_.strip())
    #         try:
    #             reactants=Chem.CanonSmiles('.'.join(temp))
    #             products=Chem.CanonSmiles(prods_raw[idx])
    #             rcts.append(reactants)
    #             prods.append(products)
    #             unproc_smile.append(unproc_smiles[idx])
    #         except:
    #             fail.append('.'.join(temp)+'>>'+prods_raw[idx])
    #             continue
    #     assert len(rcts)==len(prods)
    #     ArgumentParser.validate_translate_opts(opt)
    #     logger = init_logger(opt.log_file)
    #     translator = build_translator(opt, logger=logger, report_score=True)
    #     # result=pd.DataFrame(columns=['rcts','pred_products','pred_scores','rxn_smile'])
    #     result=pd.DataFrame(columns=['rcts','actual_smile','rxn_smiles'])
    #     for idx,rct in enumerate(tqdm(rcts)):
    #         temp=pd.DataFrame()
    #         temp['rcts']=[rct]
    #         # temp['classes']=[df.iloc[idx]['classes']]
    #         # res=get_products(rct,translator,opt)
    #         # # print(res[1])
    #         # # print(res[0])
    #         # temp['pred_scores']=[res[1]]
    #         # temp['pred_products']=[res[0]]
    #         temp['actual_smile']=unproc_smile[idx]
    #         temp['rxn_smiles']=[Chem.CanonSmiles(rct)+'>>'+prods[idx]]
    #         result=pd.concat([result,temp])
    #         file_name_=file_name.split('.')[0]
    #         # result.to_pickle(f'/home/ubuntu/joel/manual_reactions/fwd_results/{file_name_}_base.pickle')
    #         result.to_csv(f'/home/ubuntu/joel/manual_reactions/fwd_results/{file_name_}_succ.csv')
    #         fail_df=pd.DataFrame()
    #         fail_df['rxn_smiles']=fail
    #         fail_df.to_csv(f'/home/ubuntu/joel/manual_reactions/fwd_results/{file_name_}_fail.csv')

if __name__ == "__main__":
    main()