# ─────────────────────────────────────────────────────────────────────────────
# LICENSE
# ─────────────────────────────────────────────────────────────────────────────
# From the ACI-BENCH repository by Yim et al. (2023)
# Original source: https://github.com/wyim/aci-bench
# Published under Creative Commons Attribution 4.0 International License (CC BY 4.0)
# https://creativecommons.org/licenses/by/4.0/

# ─────────────────────────────────────────────────────────────────────────────
# CITATION
# ─────────────────────────────────────────────────────────────────────────────
# @article{aci-bench,
#   author = {Wen{-}wai Yim and
#             Yujuan Fu and
#             Asma {Ben Abacha} and
#             Neal Snider and Thomas Lin and Meliha Yetisgen},
#   title = {ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for
#            Benchmarking Automatic Visit Note Generation},
#   journal = {Nature Scientific Data},
#   year = {2023}
# }

# ─────────────────────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────────────────────
# Must be run from the project root (PythonProject4/).
#
# Example for each experiment:
#   python evaluation/evaluate_fullnote.py \
#       data/clinicalnlp_taskB_test1.csv \
#       results/aci_bench_style_baseline_output/gvr_results_predictions.csv \
#       data/clinicalnlp_taskB_test1_metadata.csv

#   python evaluation/evaluate_fullnote.py \
#       data/clinicalnlp_taskB_test1.csv \
#       results/aci_bench_style_baseline_output/cd_results_predictions.csv \
#       data/clinicalnlp_taskB_test1_metadata.csv

#   python evaluation/evaluate_fullnote.py \
#       data/clinicalnlp_taskB_test1.csv \
#       results/aci_bench_style_baseline_output/pe_results_predictions.csv \
#       data/clinicalnlp_taskB_test1_metadata.csv
#
# Arguments:
#   <gold>           path to the gold CSV (clinicalnlp_taskB_test1.csv)
#   <sys>            path to the predictions CSV
#                    (e.g. results/aci_bench_style_baseline_output/gvr_results_predictions.csv)
#   <metadata-file>  path to the metadata CSV (clinicalnlp_taskB_test1_metadata.csv)
#
# Output:
#   results/<predictions_filename>.json  — full breakdown by subset, division,
#                                          transcript length, and metadata
#
# Note: rougeLsum is the summary-level ROUGE-L used in the ACI-Bench paper.
#       rougeL (sentence-level) is also stored in the JSON but is not the paper metric.
#       MEDCON (UMLS) scoring is currently stubbed out (returns 0.0).
#       To enable: conda activate umls_env before running.

# ─────────────────────────────────────────────────────────────────────────────
# INSIGNIFICANT MODIFICATIONS BY TEAM 26 (CS6180, Northeastern University, 2026)
# ─────────────────────────────────────────────────────────────────────────────
#   - Added os.makedirs to auto-create results/ directory if it doesn't exist
#   - Stubbed out BERTScore and BLEURT pending Python 3.13 compatibility fix
#   - Replaced wildcard UMLS import with explicit import
#   - Fixed terminal output to print rougeLsum instead of rougeL
#   - Removed CUDA/torch dependency (not applicable on M1 Mac)
#   - Added UMLS stub function as default for environments without QuickUMLS

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

# standard library imports
import sys
import json
import os  # needed for os.makedirs — auto-creates results/ dir if it doesn't exist

# third-party imports
import evaluate
import pandas as pd
import numpy as np
# import torch  # original uses CUDA GPU; not needed on M1 Mac

# local imports
from UMLS_evaluation import umls_score_group
from sectiontagger import SectionTagger

# ─────────────────────────────────────────────────────────────────────────────
# UMLS STUB — Team 26 addition
# ─────────────────────────────────────────────────────────────────────────────
# Default stub so file runs without QuickUMLS configured.
# To enable real MEDCON scoring: comment out this function and ensure
# umls_env is activated before running.
def umls_score_group(references, predictions, use_umls=True):
    """Stub: returns zeros until QuickUMLS is configured."""
    return [0.0] * len(references)


# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

section_tagger = SectionTagger()

SECTION_DIVISIONS = [ 'subjective', 'objective_exam', 'objective_results', 'assessment_and_plan' ]

def read_text( fn ) :
    texts = None
    if ".csv" in fn:
        temp_df=pd.read_csv(fn)
        texts=[temp_df["note"][ind] for ind in range(len(temp_df))]
    else:
        with open( fn ) as f :
            texts = f.readlines()
    return texts


def add_section_divisions( row ) :
    row[ 'src_len' ] = len( row[ 'dialogue' ].split() )
    for evaltype in [ 'reference', 'prediction' ] :
        text = row[ evaltype ]
        text_with_endlines = text.replace( '__lf1__', '\n' )
        detected_divisions = section_tagger.divide_note_by_metasections( text_with_endlines )
        for detected_division in detected_divisions :
            label, _, _, start, _, end = detected_division
            row[ '%s_%s' %( evaltype, label ) ] = text_with_endlines[ start:end ].replace( '\n', '__lf1__' )
    return row


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

if len( sys.argv ) < 3 :
    print( 'usage: python evaluate_fullnote.py <gold> <sys> <metadata-file>' )
    sys.exit(0)

fn_gold = sys.argv[1]
fn_sys = sys.argv[2]
fn_metadata = [ sys.argv[3] if len( sys.argv )>3 else None ][0]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

# read in reference/hyp files
references = read_text( fn_gold )
predictions = read_text( fn_sys  )
# handle edge cases
predictions = [str(s) for s in predictions]
print( 'gold path: %s [%s summuaries]' %(fn_gold, len(references) ) )
print( 'system path: %s [%s summuaries]' %(fn_sys, len(predictions) ) )

# read in metadata file - if none exists, just creates a dummy
if fn_metadata :
    df = pd.read_csv( fn_metadata )
    df[ 'reference' ] = references
    df[ 'prediction' ] = predictions
    df[ 'dialogue' ] = pd.read_csv(fn_gold)['dialogue']
else :
    data = [ { 'id':ind, 'dataset':0, 'dialogue':'', 'reference':references[ind], 'prediction':predictions[ind]  } for ind in range( len( references ) ) ]
    df = pd.DataFrame( data )

# add section divisions
df = df.apply( lambda row: add_section_divisions( row ), axis=1 )
# fill in missing section divisions as empty string
df = df.fillna( '#####EMPTY#####' )
num_test = len( df )

# ─────────────────────────────────────────────────────────────────────────────
# BUILD EVALUATION LISTS
# ─────────────────────────────────────────────────────────────────────────────
# Append each division's text to the references/predictions lists so all
# metrics can be computed in a single pass. After this loop the lists are
# 5x the original length: [full_note x N, subjective x N, objective_exam x N,
# objective_results x N, assessment_and_plan x N]

for division in SECTION_DIVISIONS :
    references.extend( df.get( 'reference_%s' %( division ), ['']*num_test ) )
    predictions.extend( df.get( 'prediction_%s' %( division ), ['']*num_test ) )

# sanity check: should now have 5x the original set (full note + 4 divisions)
assert len( references ) == len(df)*5, 'The number of expected references does not match expected'
assert len( predictions ) == len(df)*5, 'The number of expected predictions does not match expected'

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────

results_umls_all = umls_score_group(references, predictions)
results_NER_all  = umls_score_group(references, predictions, False)

results_rouge_all = evaluate.load('rouge').compute(
    references=references, predictions=predictions, use_aggregator=False
)

# TODO: restore when Python 3.13/DeBERTa tokenizer compatibility is resolved
# results_bertscore = evaluate.load('bertscore').compute(
#     references=references, predictions=predictions, model_type='microsoft/deberta-xlarge-mnli'
# )
results_bertscore = {
    'precision': [0.0] * len(references),
    'recall':    [0.0] * len(references),
    'f1':        [0.0] * len(references),
}

# TODO: restore when Python 3.13/DeBERTa tokenizer compatibility is resolved
# results_bleurt = evaluate.load('bleurt', config_name='BLEURT-20').compute(
#     references=references, predictions=predictions
# )
results_bleurt = {'scores': [0.0] * len(references)}

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

results_all = {
    "num_test": num_test,
    'ALL': {
        'rouge1':             np.mean( results_rouge_all['rouge1'][:num_test] ),
        'rouge2':             np.mean( results_rouge_all['rouge2'][:num_test] ),
        'rougeL':             np.mean( results_rouge_all['rougeL'][:num_test] ),
        'rougeLsum':          np.mean( results_rouge_all['rougeLsum'][:num_test] ),
        'bertscore-precision':np.mean( results_bertscore['precision'][:num_test] ),
        'bertscore-recall':   np.mean( results_bertscore['recall'][:num_test] ),
        'bertscore-f1':       np.mean( results_bertscore['f1'][:num_test] ),
        'bleurt':             np.mean( results_bleurt['scores'][:num_test] ),
        'umls':               np.mean( results_umls_all[:num_test] ),
        'NER':                np.mean( results_NER_all[:num_test] ),
    }
}

######## CALCULATE PER-SUBSET SCORES ########
def select_values_by_indices( lst, indices ) :
    return [ lst[ind] for ind in indices ]

subsets = df[ 'dataset' ].unique().tolist()
for subset in subsets :
    indices = df[ df['dataset']==subset ].index.tolist()
    results_all[ 'dataset-%s' %subset ] = {
        'rouge1':              np.mean( select_values_by_indices( results_rouge_all['rouge1'][:num_test], indices ) ),
        'rouge2':              np.mean( select_values_by_indices( results_rouge_all['rouge2'][:num_test], indices ) ),
        'rougeL':              np.mean( select_values_by_indices( results_rouge_all['rougeL'][:num_test], indices ) ),
        'rougeLsum':           np.mean( select_values_by_indices( results_rouge_all['rougeLsum'][:num_test], indices ) ),
        'bertscore-precision': np.mean( select_values_by_indices( results_bertscore['precision'][:num_test], indices ) ),
        'bertscore-recall':    np.mean( select_values_by_indices( results_bertscore['recall'][:num_test], indices ) ),
        'bertscore-f1':        np.mean( select_values_by_indices( results_bertscore['f1'][:num_test], indices ) ),
        'bleurt':              np.mean( select_values_by_indices( results_bleurt['scores'][:num_test], indices ) ),
        'umls':                np.mean( select_values_by_indices( results_umls_all[:num_test], indices ) ),
        'NER':                 np.mean( select_values_by_indices( results_NER_all[:num_test], indices ) ),
    }

######## CALCULATE PER-DIVISION SCORES ########
for ind, division in enumerate( SECTION_DIVISIONS ) :
    start = (ind+1) * num_test
    end   = (ind+2) * num_test
    results_all[ 'division-%s' %division ] = {
        'rouge1':              np.mean( results_rouge_all['rouge1'][start:end] ),
        'rouge2':              np.mean( results_rouge_all['rouge2'][start:end] ),
        'rougeL':              np.mean( results_rouge_all['rougeL'][start:end] ),
        'rougeLsum':           np.mean( results_rouge_all['rougeLsum'][start:end] ),
        'bertscore-precision': np.mean( results_bertscore['precision'][start:end] ),
        'bertscore-recall':    np.mean( results_bertscore['recall'][start:end] ),
        'bertscore-f1':        np.mean( results_bertscore['f1'][start:end] ),
        'bleurt':              np.mean( results_bleurt['scores'][start:end] ),
        'umls':                np.mean( results_umls_all[start:end] ),
        'NER':                 np.mean( results_NER_all[start:end] ),
    }

######## CALCULATE PER-LENGTH SCORES (bigger than 512 vs not) ########
df_shortsrc = df[ df['src_len']<=512 ]
if len( df_shortsrc ) > 0 :
    indices = df_shortsrc[:num_test].index.tolist()
    results_all[ 'shorter-src' ] = {
        'rouge1':              np.mean( select_values_by_indices( results_rouge_all['rouge1'][:num_test], indices ) ),
        'rouge2':              np.mean( select_values_by_indices( results_rouge_all['rouge2'][:num_test], indices ) ),
        'rougeL':              np.mean( select_values_by_indices( results_rouge_all['rougeL'][:num_test], indices ) ),
        'rougeLsum':           np.mean( select_values_by_indices( results_rouge_all['rougeLsum'][:num_test], indices ) ),
        'bertscore-precision': np.mean( select_values_by_indices( results_bertscore['precision'][:num_test], indices ) ),
        'bertscore-recall':    np.mean( select_values_by_indices( results_bertscore['recall'][:num_test], indices ) ),
        'bertscore-f1':        np.mean( select_values_by_indices( results_bertscore['f1'][:num_test], indices ) ),
        'bleurt':              np.mean( select_values_by_indices( results_bleurt['scores'][:num_test], indices ) ),
        'umls':                np.mean( select_values_by_indices( results_umls_all[:num_test], indices ) ),
        'NER':                 np.mean( select_values_by_indices( results_NER_all[:num_test], indices ) ),
    }

df_longsrc = df[ df['src_len']>512 ]
if len( df_longsrc ) > 0 :
    indices = df_longsrc[:num_test].index.tolist()
    results_all[ "longer-src (support:{})".format(len(df_longsrc)) ] = {
        'rouge1':              np.mean( select_values_by_indices( results_rouge_all['rouge1'][:num_test], indices ) ),
        'rouge2':              np.mean( select_values_by_indices( results_rouge_all['rouge2'][:num_test], indices ) ),
        'rougeL':              np.mean( select_values_by_indices( results_rouge_all['rougeL'][:num_test], indices ) ),
        'rougeLsum':           np.mean( select_values_by_indices( results_rouge_all['rougeLsum'][:num_test], indices ) ),
        'bertscore-precision': np.mean( select_values_by_indices( results_bertscore['precision'][:num_test], indices ) ),
        'bertscore-recall':    np.mean( select_values_by_indices( results_bertscore['recall'][:num_test], indices ) ),
        'bertscore-f1':        np.mean( select_values_by_indices( results_bertscore['f1'][:num_test], indices ) ),
        'bleurt':              np.mean( select_values_by_indices( results_bleurt['scores'][:num_test], indices ) ),
        'umls':                np.mean( select_values_by_indices( results_umls_all[:num_test], indices ) ),
        'NER':                 np.mean( select_values_by_indices( results_NER_all[:num_test], indices ) ),
    }

######## CALCULATE PER-METADATA SCORES ########
for meta_type in ["patient_gender", "cc", "2nd_complaints"]:
    if meta_type in df:
        values = set(list(df[meta_type]))
        for value in values:
            df_meta = df[ df[meta_type] == value ]
            if len( df_meta ) > 0 :
                indices = df_meta[:num_test].index.tolist()
                results_all[ meta_type+"-{}(support:{})".format(value, len(df_meta)) ] = {
                    'rouge1':              np.mean( select_values_by_indices( results_rouge_all['rouge1'][:num_test], indices ) ),
                    'rouge2':              np.mean( select_values_by_indices( results_rouge_all['rouge2'][:num_test], indices ) ),
                    'rougeL':              np.mean( select_values_by_indices( results_rouge_all['rougeL'][:num_test], indices ) ),
                    'rougeLsum':           np.mean( select_values_by_indices( results_rouge_all['rougeLsum'][:num_test], indices ) ),
                    'bertscore-precision': np.mean( select_values_by_indices( results_bertscore['precision'][:num_test], indices ) ),
                    'bertscore-recall':    np.mean( select_values_by_indices( results_bertscore['recall'][:num_test], indices ) ),
                    'bertscore-f1':        np.mean( select_values_by_indices( results_bertscore['f1'][:num_test], indices ) ),
                    'bleurt':              np.mean( select_values_by_indices( results_bleurt['scores'][:num_test], indices ) ),
                    'umls':                np.mean( select_values_by_indices( results_umls_all[:num_test], indices ) ),
                    'NER':                 np.mean( select_values_by_indices( results_NER_all[:num_test], indices ) ),
                }

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs('../results', exist_ok=True)
json_object = json.dumps( results_all, indent=4 )
fn_out = 'results/new{}.json'.format(fn_sys.split("/")[-1].split(".")[0])
with open( fn_out, 'w' ) as f :
    f.write( json_object )

print(f"\nResults saved to: {fn_out}")
print(f"\n=== FULL NOTE SCORES ===")
print(f"ROUGE-1:       {results_all['ALL']['rouge1']:.4f}")
print(f"ROUGE-2:       {results_all['ALL']['rouge2']:.4f}")
print(f"ROUGE-Lsum:    {results_all['ALL']['rougeLsum']:.4f}  (summary-level, matches ACI-Bench paper)")
print(f"UMLS (MEDCON): {results_all['ALL']['umls']:.4f}")
print(f"BERTScore-F1:  {results_all['ALL']['bertscore-f1']:.4f}  (stubbed — 0.0 until Python 3.13 fix)")
print(f"BLEURT:        {results_all['ALL']['bleurt']:.4f}  (stubbed — 0.0 until Python 3.13 fix)")