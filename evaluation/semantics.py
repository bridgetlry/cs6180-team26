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
# used in UMLS_evaluation.py

# ─────────────────────────────────────────────────────────────────────────────
# MODIFICATIONS BY TEAM 26 (CS6180, Northeastern University, 2026)
# ─────────────────────────────────────────────────────────────────────────────
# NONE

#source https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemGroups_2018.txt
import os
PROJECT_DIR = os.path.dirname( os.path.dirname( __file__ ) )

lines=open(PROJECT_DIR+"/resources/semantic_types.txt").read().split('\n')
types_to_exclude=["Procedures","Organizations","Occupations","Living Beings","Objects","Geographic Areas","Concepts & Ideas","Activities & Behaviors"]
SEMANTICS,types_included=[],[]
id_to_exclude=[]
#types included:
#['Phenomena', 'Devices', 'Genes & Molecular Sequences', 'Disorders', 'Physiology', 'Anatomy', 'Chemicals & Drugs']
for line in lines:
    if line:
        type,id = line.split("|")[1:3]
        if type not in types_to_exclude and id not in id_to_exclude:
            types_included.append(type)
            SEMANTICS.append(id)

# print(SEMANTICS)
# types_included=list(set(types_included))
# print(types_included)