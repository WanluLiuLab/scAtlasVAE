#!/usr/bin/env python
# coding: utf-8

# # Import Basic Packages and Definitions

# ```
# scanpy==1.8.2
# anndata=0.8.0
# numpy==1.20.3
# pandas==1.5.3
# ```

# In[17]:


import scanpy as sc
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import gc

def setPltLinewidth(linewidth:float):
    matplotlib.rcParams['axes.linewidth'] = linewidth
    
setPltLinewidth(1)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.family'] = "Arial"


# In[6]:


from colour import Color
from matplotlib.colors import LinearSegmentedColormap
import json
import pandas as pd 
import numpy as np
from colors import *

print("Loading Definitions...", end="")
def FLATTEN(x): return [i for s in x for i in s]

def createFig(figsize=(8, 4)):
    fig,ax=plt.subplots()           
    ax.spines['right'].set_color('none')     
    ax.spines['top'].set_color('none')
    #ax.spines['bottom'].set_color('none')     
    #ax.spines['left'].set_color('none')
    for line in ax.yaxis.get_ticklines():
        line.set_markersize(5)
        line.set_color("#585958")
        line.set_markeredgewidth(0.5)
    for line in ax.xaxis.get_ticklines():
        line.set_markersize(5)
        line.set_markeredgewidth(0.5)
        line.set_color("#585958")
    ax.set_xbound(0,10)
    ax.set_ybound(0,10)
    fig.set_size_inches(figsize)
    return fig,ax

def createSubplots(nrow,ncol, figsize=(8,8)):
    fig,axes=plt.subplots(nrow, ncol)
    for ax in axes.flatten():
        ax.spines['right'].set_color('none')     
        ax.spines['top'].set_color('none')
        for line in ax.yaxis.get_ticklines():
            line.set_markersize(5)
            line.set_color("#585958")
            line.set_markeredgewidth(0.5)
        for line in ax.xaxis.get_ticklines():
            line.set_markersize(5)
            line.set_markeredgewidth(0.5)
            line.set_color("#585958")
    fig.set_size_inches(figsize)
    return fig,axes

def loadHLAGenotype():
    with open('./data/genotype.json', "r") as f:
        genotype = json.load(f)
    with open('./data/partial_genotype.json', "r") as f:
        partial_genotype = json.load(f)
    return genotype, partial_genotype

MHCI_DEFINITION = ['A_1', 'A_2', 'B_1', 'B_2', 'C_1', 'C_2']

genotype, partial_genotype = loadHLAGenotypeAsPandas()
partial_genotype_raw = partial_genotype.copy()
partial_genotype = partial_genotype.loc[
    np.array(partial_genotype["A_1"] != '-') &
    np.array(partial_genotype["A_2"] != '-') &
    np.array(partial_genotype["B_1"] != '-') &
    np.array(partial_genotype["B_2"] != '-') &
    np.array(partial_genotype["C_1"] != '-') &
    np.array(partial_genotype["C_2"] != '-')
]

for i in partial_genotype.columns:
    partial_genotype[i] = list(
        map(lambda x: x.split('*')[0] + ':'.join(x.split('*')[-1].split(":")[:1]), 
            partial_genotype[i]
        ))

individual2hla = dict(list(map(lambda x: (sample2individual[x[0]],x[1]), filter(lambda x: x[0] in sample2individual.keys(), zip(partial_genotype.index, partial_genotype.to_numpy())))))
    

def to_pairing_level(i):
    if i < 11:
        return i
    elif i < 25:
        return 25
    elif i < 50:
        return 50
    elif i < 100:
        return 100
    elif i < 250:
        return 250
    elif i < 750:
        return 750
    else:
        return 751
    


def rgb2hex(vals, rgbtype=1):
    """
    Converts RGB values in a variety of formats to Hex values.

    @param  vals (tuple)     An RGB/RGBA tuple
    @param  rgbtype (int)    Valid valus are:
                        1 - Inputs are in the range 0 to 1
                        256 - Inputs are in the range 0 to 255

    @return A hex string in the form '#RRGGBB' or '#RRGGBBAA'
    """

    if len(vals) != 3 and len(vals) != 4:
        raise Exception(
            "RGB or RGBA inputs to rgb2hex must have three or four elements!")
    if rgbtype != 1 and rgbtype != 256:
        raise Exception("rgbtype must be 1 or 256!")

    # Convert from 0-1 RGB/RGBA to 0-255 RGB/RGBA
    if rgbtype == 1:
        vals = [255*x for x in vals]

    # Ensure values are rounded integers, convert to hex, and concatenate
    return '#' + ''.join(['{:02X}'.format(int(round(x))) for x in vals])
    
    
def onemer(s):
    if type(s) == str:
        return list(s)
    else:
        return ''
    
def twomer(s):
    if type(s) != str:
        return ''
    if len(s) == 1:
        return [s]
    ret = []
    for i in range(len(s)-1):
        ret.append(s[i:i+2])
    return ret
    
def make_colormap( colors, show_palette = False ): 
    color_ramp = LinearSegmentedColormap.from_list( 'my_list', [ Color( c1 ).rgb for c1 in colors ] )
    if show_palette:
        plt.figure( figsize = (15,3))
        plt.imshow( [list(np.arange(0, len( colors ) , 0.1)) ] , interpolation='nearest', origin='lower', cmap= color_ramp )
        plt.xticks([])
        plt.yticks([])
    return color_ramp

kw = dict(arrowprops=dict(arrowstyle="-"), zorder=0, va="center")

def custom_pie2(ax,anno,cm_dict,radius=1,width=1,setp=False,show_anno=False):
    pie,_=ax.pie(anno.values(), 
                 radius=radius,
                 colors = [cm_dict[p] for p in anno.keys()], 
                 # wedgeprops=dict(width=width, edgecolor='w')
                )
    if setp:
        plt.setp(pie, width = width, edgecolor='w')
    for i, p in enumerate(pie):
        theta1, theta2 = p.theta1, p.theta2
        center, r = p.center, p.r
        ang = (p.theta2 - p.theta1)/2. + p.theta1 
        x = r * np.cos(np.pi / 180 * (theta1+theta2)/2) + center[0]
        y = r * np.sin(np.pi / 180 * (theta1+theta2)/2) + center[1]
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))] 
        connectionstyle = "arc3, rad=0"
        kw["arrowprops"].update({"connectionstyle": connectionstyle}) 
        percentage = anno[list(anno.keys())[i]] / sum(list(anno.values()))
        if show_anno:
            if anno[list(anno.keys())[i]] / sum(list(anno.values())) > 0.005: 
                ax.annotate(list(anno.keys())[i] + ", " + str(round(percentage * 100,2)) + "%", xy=(x,y), xytext=(x*1.2, y*1.2), horizontalalignment=horizontalalignment, size=6, fontweight=100, color=cm_dict[list(anno.keys())[i]]) 
    # ax.text(x=0,y=0,s=str(sum(anno.values())),size=6, fontweight=900,horizontalalignment="center") 
    return pie



CELL_TYPE_CMAP = {
    "CD8": make_colormap(['#ffffff','#ffca39']),
    'CD4': make_colormap(['#ffffff','#1c83c5']),
    'Treg': make_colormap(['#ffffff','#6a4d93']),
    'MAIT': make_colormap(['#ffffff',"#2a9d8f"]),
    "cd8": make_colormap(['#ffffff','#ffca39']),
    'cd4': make_colormap(['#ffffff','#1c83c5']),
    'treg': make_colormap(['#ffffff','#6a4d93']),
    'mait': make_colormap(['#ffffff',"#2a9d8f"])
}

CELL_TYPE_C = {
    "CD8": '#ffca39',
    'CD4': '#1c83c5',
    'Treg': '#6a4d93',
    'MAIT': "#2a9d8f",
    "cd8": '#ffca39',
    'cd4': '#1c83c5',
    'treg': '#6a4d93',
    'mait': "#2a9d8f",
    'DP/DN': "#F7F7F7",
    'Unknown': "#F7F7F7",
    'Undefined': "#F7F7F7",
    'Unpredictive': "#F7F7F7",
}

CELL_TYPE_SHIFT = {
    "CD8": (-0.5,0.5),
    'CD4': (0.5,0.5),
    'Treg': (-0.5,-0.5),
    'MAIT': (0.5,-0.5),
    "cd8": (-0.5,0.5),
    'cd4': (0.5,0.5),
    'treg': (-0.5,-0.5),
    'mait': (0.5,-0.5)
}

TRBV_SEQUENCE = {
    'TRBV1': 'CTSSQ',
    'TRBV2': 'CASSE',
    'TRBV3-1': 'CASSQ',
    'TRBV3-2': 'CASSQ',
    'TRBV4-1': 'CASSQ',
    'TRBV4-2': 'CASSQ',
    'TRBV4-3': 'CASSQ',
    'TRBV5-1': 'CASSL',
    'TRBV5-3': 'CARSL',
    'TRBV5-4': 'CASSL',
    'TRBV5-5': 'CASSL',
    'TRBV5-6': 'CASSL',
    'TRBV5-7': 'CASSL',
    'TRBV5-8': 'CASSL',
    'TRBV6-1': 'CASSE',
    'TRBV6-2': 'CASSY',
    'TRBV6-3': 'CASSY',
    'TRBV6-4': 'CASSD',
    'TRBV6-5': 'CASSY',
    'TRBV6-6': 'CASSY',
    'TRBV6-7': 'CASSY',
    'TRBV6-8': 'CASSY',
    'TRBV6-9': 'CASSY',
    'TRBV7-1': 'CASSS',
    'TRBV7-2': 'CASSL',
    'TRBV7-3': 'CASSL',
    'TRBV7-4': 'CASSL',
    'TRBV7-6': 'CASSL',
    'TRBV7-7': 'CASSL',
    'TRBV7-8': 'CASSL',
    'TRBV7-9': 'CASSL',
    'TRBV9': 'CASSV',
    'TRBV10-1': 'CASSE',
    'TRBV10-2': 'CASSE',
    'TRBV10-3': 'CAISE',
    'TRBV11-1': 'CASSL',
    'TRBV11-2': 'CASSL',
    'TRBV11-3': 'CASSL',
    'TRBV12-1': 'CASSF',
    'TRBV12-2': 'CASRL',
    'TRBV12-3': 'CASSL',
    'TRBV12-4': 'CASSL',
    'TRBV12-5': 'CASGL',
    'TRBV13': 'CASSL',
    'TRBV14': 'CASSQ',
    'TRBV15': 'CATSR',
    'TRBV16': 'CASSQ',
    'TRBV17': 'YSSG',
    'TRBV18': 'CASSP',
    'TRBV19': 'CASSI',
    'TRBV20': 'CSAR',
    'TRBV20OR9-2': 'CSAR', #???
    'TRBV20-1': 'CSAR',
    'TRBV21-1': 'CASSK',
    'TRBV23-1': 'CASSQ',
    'TRBV24-1': 'CATSDL',
    'TRBV25-1': 'CASSE',
    'TRBV26': 'YASSS',
    'TRBV27': 'CASSL',
    'TRBV28': 'CASSL',
    'TRBV29-1': 'CSVE',
    'TRBV30': 'CAWS'
}


TRBJ_SEQUENCE = {
    'TRBJ1-1': 'NTEAFFGQGTRLTVV',
    'TRBJ1-2': 'NYGYTFGSGTRLTVV',
    'TRBJ1-3': 'SGNTIYFGEGSWLTVV',
    'TRBJ1-4': 'TNEKLFFGSGTQLSVL',
    'TRBJ1-5': 'SNQPQHFGDGTRLSIL',
    'TRBJ1-6': 'SYNSPLHFGNGTRLTVT',
    'TRBJ2-1': 'SYNEQFFGPGTRLTVL',
    'TRBJ2-2': 'NTGELFFGEGSRLTVL',
    'TRBJ2-3': 'STDTQYFGPGTRLTVL',
    'TRBJ2-4': 'AKNIQYFGAGTRLSVL',
    'TRBJ2-5': 'QETQYFGPGTRLLVL',
    'TRBJ2-6': 'SGANVLTFGAGSRLTVL',
    'TRBJ2-7': 'SYEQYFGPGTRLTVT'
}
TRAV_SEQUENCE = {
'TRAV1-1': 'CAVR',
'TRAV1-2': 'CAVR',
'TRAV2': 'CAVE',
'TRAV3': 'CAVRD',
'TRAV4': 'CLVGD',
'TRAV5': 'CAES',
'TRAV6': 'CALD',
'TRAV7': 'CAVD',
'TRAV8-1': 'CAVN',
'TRAV8-2': 'CVVS',
'TRAV8-3': 'CAVG',
'TRAV8-4': 'CAVS',
'TRAV8-6': 'CAVS',
'TRAV8-7': 'CAVG',
'TRAV9-1': 'CALS',
'TRAV9-2': 'CALS',
'TRAV10': 'CVVS',
'TRAV11': 'CAL',
'TRAV12-1': 'CVVN',
'TRAV12-2': 'CAVN',
'TRAV12-3': 'CAMS',
'TRAV13-1': 'CAAS',
'TRAV13-2': 'CAEN',
'TRAV14DV4': 'CAMRE',
'TRAV16': 'CALS',
'TRAV17': 'CATD',
'TRAV18': 'CALR',
'TRAV19': 'CALSE',
'TRAV20': 'CAVQ',
'TRAV21': 'CAVR',
'TRAV22': 'CAVE',
'TRAV23DV6': 'CAAS',
'TRAV24': 'CAF',
'TRAV25': 'CAG',
'TRAV26-1': 'CIVRV',
'TRAV26-2': 'CILRD',
'TRAV27': 'CAG',
'TRAV29DV5': 'CAAS',
'TRAV30': 'CGTE',
'TRAV34': 'CGAD',
'TRAV35': 'CAGQ',
'TRAV36DV7': 'CAVE',
'TRAV38-1': 'CAFMK',
'TRAV38-2DV8': 'CAYRS',
'TRAV39': 'CAVD',
'TRAV40': 'CLLG',
'TRAV41': 'CAVR'
}


TRAJ_SEQUENCE = {
    'TRAJ1': 'YESITSQLQFGKGTRVSTSP',
    'TRAJ2': 'NTGGTIDKLTFGKGTHVFIIS',
    'TRAJ3': 'GYSSASKIIFGSGTRLSIRP',
    'TRAJ4': 'FSGGYNKLIFGAGTRLAVHP',
    'TRAJ5': 'DTGRRALTFGSGTRLQVQP',
    'TRAJ6': 'ASGGSYIPTFGRGTSLIVHP',
    'TRAJ7': 'DYGNNRLAFGKGNQVVVIP',
    'TRAJ8': 'NTGFQKLVFGTGTRLLVSP',
    'TRAJ9': 'GNTGGFKTIFGAGTRLFVKA',
    'TRAJ10': 'ILTGGGNKLTFGTGTQLKVEL',
    'TRAJ11': 'NSGYSTLTFGKGTMLLVSP',
    'TRAJ12': 'MDSSYKLIFGSGTRLLVRP',
    'TRAJ13': 'NSGGYQKVTFGIGTKLQVIP',
    'TRAJ14': 'IYSTFIFGSGTRLSVKP',
    'TRAJ15': 'NQAGTALIFGKGTTLSVSS',
    'TRAJ16': 'FSDGQKLLFARGTMLKVDL',
    'TRAJ17': 'IKAAGNKLTFGGGTRVLVKP',
    'TRAJ18': 'DRGSTLGRLYFGRGTQLTVWP',
    'TRAJ19': 'YQRFYNFTFGKGSKHNVTP',
    'TRAJ20': 'SNDYKLSFGAGTTVTVRA',
    'TRAJ21': 'YNFNKFYFGSGTKLNVKP',
    'TRAJ22': 'SSGSARQLTFGSGTQLTVLP',
    'TRAJ23': 'IYNQGGKLIFGQGTELSVKP',
    'TRAJ24': 'TTDSWGKFEFGAGTQVVVTP',
    'TRAJ25': 'EGQGFSFIFGKGTRLLVKP',
    'TRAJ26': 'DNYGQNFVFGPGTRLSVLP',
    'TRAJ27': 'NTNAGKSTFGDGTTLTVKP',
    'TRAJ28': 'YSGAGSYQLTFGKGTKLSVIP',
    'TRAJ29': 'NSGNTPLVFGKGTRLSVIA',
    'TRAJ30': 'NRDDKIIFGKGTRLHILP',
    'TRAJ31': 'NNNARLMFGDGTQLVVKP',
    'TRAJ32': 'NYGGATNKLIFGTGTLLAVQP',
    'TRAJ33': 'DSNYQLIWGAGTKLIIKP',
    'TRAJ34': 'SYNTDKLIFGTGTRLQVFP',
    'TRAJ35': 'IGFGNVLHCGSGTQVIVLP',
    'TRAJ36': 'QTGANNLFFGTGTRLTVIP',
    'TRAJ37': 'GSGNTGKLIFGQGTTLQVKP',
    'TRAJ38': 'NAGNNRKLIWGLGTSLAVNP',
    'TRAJ39': 'NNNAGNMLTFGGGTRLMVKP',
    'TRAJ40': 'TTSGTYKYIFGTGTRLKVLA',
    'TRAJ41': 'NSNSGYALNFGKGTSLLVTP',
    'TRAJ42': 'NYGGSQGNLIFGKGTKLSVKP',
    'TRAJ43': 'NNNDMRFGAGTRLTVKP',
    'TRAJ44': 'NTGTASKLTFGTGTRLQVTL',
    'TRAJ45': 'YSGGGADGLTFGKGTHLIIQP',
    'TRAJ46': 'KKSSGDKLTFGTGTRLAVRP',
    'TRAJ47': 'EYGNKLVFGAGTILRVKS',
    'TRAJ48': 'SNFGNEKLTFGTGTRLTIIP',
    'TRAJ49': 'NTGNQFYFGTGTSLTVIP',
    'TRAJ50': 'KTSYDKVIFGPGTSLSVIP',
    'TRAJ51': 'MRDSYEKLIFGKET*LTVKP',
    'TRAJ52': 'NAGGTSYGKLTFGQGTILTVHP',
    'TRAJ53': 'NSGGSNYKLTFGKGTLLTVNP',
    'TRAJ54': 'IQGAQKLVFGQGTRLTINP',
    'TRAJ55': 'KCW*CSCWGKGMSTKINP',
    'TRAJ56': 'YTGANSKLTFGKGITLSVRP',
    'TRAJ57': 'TQGGSEKLVFGKGTKLTVNP',
    'TRAJ58': '*ETSGSRLTFGEGTQLTVNP',
    'TRAJ59': 'KEGNRKFTFGMGTQVRVKL',
    'TRAJ60': 'KIT*MLNFGKGTELIVSL',
    'TRAJ61': 'YRVNRKLTFGANTRGIMKL',
}

CELL_TYPE_DEFINITION = {'Central memory CD8 T cells': "CD8",
'Effector memory CD8 T cells': "CD8",
'Follicular helper T cells': "CD4",
'MAIT cells': "MAIT",
'Naive CD4 T cells': "CD4",
'Naive CD8 T cells': "CD8",
'T regulatory cells': "Treg",
'Terminal effector CD4 T cells': "CD4",
'Terminal effector CD8 T cells': "CD8",
'Th1 cells': "CD4",
'Th1/Th17 cells': "CD4",
'Th17 cells': "CD4",
'Th2 cells': "CD4",
'Unpredictive': "Undefined"
}


AMINO_ACIDS = ['R',
 'H',
 'K',
 'D',
 'E',
 'S',
 'T',
 'N',
 'Q',
 'C',
 'G',
 'P',
 'A',
 'V',
 'I',
 'L',
 'M',
 'F',
 'Y',
 'W'
]

AMINO_ACID_PROPERTIES = {
 'R': "Positive",
 'H': "Positive",
 'K': "Positive",
 'D': "Negative",
 'E': "Negative",
 'S': "Polar uncharged",
 'T': "Polar uncharged",
 'N': "Polar uncharged",
 'Q': "Polar uncharged",
 'C': "Special",
 'U': "Special",
 'G': "Special",
 'P': "Special",
 'A': "Hydrophobic",
 'V': "Hydrophobic",
 'I': "Hydrophobic",
 'L': "Hydrophobic",
 'M': "Hydrophobic",
 'F': "Hydrophobic",
 'Y': "Hydrophobic",
 'W': "Hydrophobic"
}

AMINO_ACIDS_IDX = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))
TRAV2IDX = dict(zip(TRAV_SEQUENCE.keys(), range(len(TRAV_SEQUENCE))))
TRAJ2IDX = dict(zip(TRAJ_SEQUENCE.keys(), range(len(TRAJ_SEQUENCE))))
TRBV2IDX = dict(zip(TRBV_SEQUENCE.keys(), range(len(TRBV_SEQUENCE))))
TRBJ2IDX = dict(zip(TRBJ_SEQUENCE.keys(), range(len(TRBJ_SEQUENCE))))
VJINDEX = {}
VJINDEX['TRAV'] = TRAV2IDX
VJINDEX['TRAJ'] = TRAJ2IDX
VJINDEX['TRBV'] = TRBV2IDX
VJINDEX['TRBJ'] = TRBJ2IDX

def imputeTRAVJsequence(cdr3a):
    maximum_v = ''
    maximum_j = ''
    maximum_vgene, maximum_vjgene = None, None
    
    for v in TRAV_SEQUENCE.keys():
        vsequence = TRAV_SEQUENCE[v]
        for m in range(len(vsequence)):
            if cdr3a.startswith(vsequence[:-(m+1)]):     
                vindex = cdr3a.index(vsequence[:-(m+1)]) + len(vsequence[:-(m+1)])
                break
        vmotif = vsequence[:-(m+1)]
        if len(vmotif) > len(maximum_v):
            maximum_v = vmotif
            maximum_vgene = v
    for j in TRAJ_SEQUENCE.keys():
    
        jsequence = TRAJ_SEQUENCE[j]
        
        if 'F' in jsequence:
            jsequence = jsequence[:len(jsequence) - jsequence[::-1].index("F")] 
        elif 'W' in jsequence:
            jsequence = jsequence[:len(jsequence) - jsequence[::-1].index("W")] 
        for n in range(len(jsequence)):    
            if cdr3a.endswith(jsequence[n:]):
                jindex = cdr3a.index(jsequence[n:])
                break 
        jmotif = jsequence[n:]
        if len(jmotif) > len(maximum_j):
            maximum_j = jmotif
            maximum_jgene = j
    return maximum_v, maximum_vgene, maximum_j, maximum_jgene

def imputeTRBVJsequence(cdr3a):
    maximum_v = ''
    maximum_j = ''
    maximum_vgene, maximum_vjgene = None, None
    
    for v in TRBV_SEQUENCE.keys():
        vsequence = TRBV_SEQUENCE[v]
        for m in range(len(vsequence)):
            if cdr3a.startswith(vsequence[:-(m+1)]):     
                vindex = cdr3a.index(vsequence[:-(m+1)]) + len(vsequence[:-(m+1)])
                break
        vmotif = vsequence[:-(m+1)]
        if len(vmotif) > len(maximum_v):
            maximum_v = vmotif
            maximum_vgene = v
    for j in TRBJ_SEQUENCE.keys():
    
        jsequence = TRBJ_SEQUENCE[j]
        
        if 'F' in jsequence:
            jsequence = jsequence[:len(jsequence) - jsequence[::-1].index("F")] 
        elif 'W' in jsequence:
            jsequence = jsequence[:len(jsequence) - jsequence[::-1].index("W")] 
        for n in range(len(jsequence)):    
            if cdr3a.endswith(jsequence[n:]):
                jindex = cdr3a.index(jsequence[n:])
                break 
        jmotif = jsequence[n:]
        if len(jmotif) > len(maximum_j):
            maximum_j = jmotif
            maximum_jgene = j
    return maximum_v, maximum_vgene, maximum_j, maximum_jgene

# response = get_response("Load computing modules?")
print(Colors.GREEN + "Success" + Colors.NC)

CDR1a = [
    'TSGFYG',
    'TSGFNG',
    'VSNAYN',
    'VSGNPY',
    'NIATNDY',
    'DSSSTY',
    'NYSPAY',
    'VSRFNN',
    'YGGTVN',
    'SSYSPS',
    'YGATPY',
    'SSVPPY',
    'SSVSVY',
    'YSGVPS',
    'TTQYPS',
    'ATGYPS',
    'VSPFSN',
    'ERTLFN',
    'NSASQS',
    'DRGSQS',
    'NSAFQY',
    'DSASNY',
    'NSASDY',
    'TSDPSYG',
    'YSGSPE',
    'TSINN',
    'SSYSTF',
    'TRDTTYY',
    'VSGLRG',
    'DSAIYN',
    'DSVNN',
    'NTAFDY',
    'SSNFYA',
    'TTLSN',
    'TISGNEY',
    'TISGTDY',
    'SVFSS',
    'NSMFDY',
    'KALYS',
    'KTLYG',
    'SIFNT',
    'VTNFRS',
    'TSENNYY',
    'TSESDYY',
    'TTSDR',
    'STGYPT',
    'VGISA'
]

CDR2a = ['NALDGL',
'NVLDGL',
'GSKP',
'YITGDNLV',
'GYKTK',
'IFSNMDM',
'IRENEKE',
'MYSAGYE',
'YFSGDPLV',
'YTSAATLV',
'YFSGDTLV',
'YTSAATLV',
'YLSGSTLV',
'DLTEATQV',
'AMKANDK',
'ATKADDK',
'MTFSENT',
'IQSSQKE',
'VYSSGN',
'IYSNGD',
'TYSSGN',
'IRSNVGE',
'IRSNMDK',
'QGSYDQQN',
'HISR',
'IRSNERE',
'SSENQE',
'RNSFDEQN',
'LYSAGEE',
'IQSSQRE',
'IPSGT',
'IRPDVSE',
'MTLNGDE',
'LVKSGEV',
'GLKNN',
'GLTSN',
'VVTGGEV',
'ISSIKDK',
'LLKGGEQ',
'LQKGGEE',
'LYKAGEL',
'LTSSGIE',
'QEAYKQQN',
'QEAYKQQN',
'LLSNGAV',
'ETME',
'LSSGK']

CDR1b = ['GHDS',
'SNHLY',
'LGHDT',
'LGHNA',
'MGHRA',
'LGHNA',
'LGHNA',
'SGHRS',
'SGHSS',
'SGHNT',
'SGHKS',
'SGHDT',
'SGHTS',
'SGHTS',
'MNHNS',
'MNHEY',
'MNHEY',
'MRHNA',
'MNHEY',
'MNHNY',
'MNHEY',
'MNHGY',
'MNHGY',
'SGHNA',
'SGHTA',
'SGHTA',
'SGHVT',
'SGHVS',
'SSHAT',
'SGHVS',
'SEHNR',
'SGDLS',
'WNHNN',
'WSHSY',
'ENHRY',
'SGHAT',
'SGHAT',
'SGHNT',
'SGHND',
'FGHNF',
'SGHNS',
'SGHDY',
'LGHNT',
'PRHDT',
'SGHDN',
'LNHNV',
'KGHSY',
'SGHMF',
'KGHSH',
'LNHDA',
'DFQATT',
'KAHSY',
'KGHTF',
'KGHDR',
'MGHDK',
'MNHVT',
'MNHEY',
'MDHEN',
'SQVTM',
'GTSNPN']

CDR2b = ['YNCKEF',
'FYNNEI',
'YNNKEL',
'YSNKEP',
'YSYEKL',
'YNFKEQ',
'YSLEER',
'YFSETQ',
'YANELR',
'YYREEE',
'YYEKEE',
'YYEEEE',
'YYEKEE',
'YDEGEE',
'SASEGT',
'SVGEGT',
'SVGEGT',
'SNTAGT',
'SVGAGI',
'SVGAGI',
'SVAAAL',
'SAAAGT',
'SVAAGI',
'FQGKDA',
'FQGNSA',
'FQGTGA',
'SQSDAQ',
'FNYEAQ',
'FNYEAQ',
'FQNEAQ',
'FQNEAQ',
'YYNGEE',
'SYGVQD',
'SAAADI',
'SYGVKD',
'FQDESV',
'FQNNGV',
'YENEEA',
'FCSWTL',
'FRS*SI',
'FNNNVP',
'FNNNVP',
'FRNRAP',
'FYEKMQ',
'FVKESK',
'YYDKDF',
'FQNENV',
'FQYQNI',
'LQKENI',
'SQIVND',
'SNEGSKA',
'FQNEEL',
'FQNEQV',
'SFDVKD',
'SYGVNS',
'SPGTGS',
'SMNVEV',
'SYDVKM',
'ANQGSEA',
'SVGIG']

TRAV2CDR1a = dict(zip(TRAV_SEQUENCE.keys(), CDR1a))
TRAV2CDR2a = dict(zip(TRAV_SEQUENCE.keys(), CDR2a))
TRBV2CDR1b = dict(zip(TRBV_SEQUENCE.keys(), CDR1b))
TRBV2CDR2b = dict(zip(TRBV_SEQUENCE.keys(), CDR2b))


# In[17]:


study_name_palette = {'Abbas_2021': '#ffff00',
 'Azizi_2018': '#1ce6ff',
 'Bacher_2020': '#ff34ff',
 'Boland_2020': '#ff4a46',
 'Borcherding_2021': '#008941',
 'Cheon_2021': '#006fa6',
 'Corridoni_2020': '#a30059',
 'Gao_2020': '#ffdbe5',
 'Gate_2020': '#7a4900',
 'He_2020': '#0000a6',
 'Kim_2022': '#63ffac',
 'Krishna_2021': '#b79762',
 'Liao_2020': '#004d43',
 'Liu_2021': '#8fb0ff',
 'Lu_2019': '#997d87',
 'Luoma_2020': '#5a0007',
 'Mahuron_2020': '#809693',
 'Neal_2018': '#6a3a4c',
 'Notarbartolo_2021': '#1b4400',
 'Penkava_2020': '#4fc601',
 'Ramaswamy_2021': '#3b5dff',
 'Simone_2021': '#4a3b53',
 'Suo_2022': '#ff2f80',
 'Wang_2021': '#61615a',
 'Wang_2022': '#ba0900',
 'Wen_2020': '#6b7900',
 'Yost_2019': '#00c2a0',
 'Zheng_2020': '#ffaa92'}

reannotated_prediction_palette = {'CD4': '#1c83c5ff',
 'CD8': '#ffca39ff',
 'CD40LG': '#5bc8d9ff',
 'Cycling': '#a7a7a7ff',
 'MAIT': '#2a9d8fff',
 'Naive CD4': '#3c3354ff',
 'Naive CD8': '#a9d55dff',
 'Treg': '#6a4d93ff',
 'Undefined': '#f7f7f7ff',
 'Ambiguous': '#f7f7f7ff',
  'Unknown': '#f7f7f7ff'}

reannotation_palette = {'CREM+ Tm': '#1f77b4',
 'CXCR6+ Tex': '#ff7f0e',
 'Cycling T': '#279e68',
 'Early Tcm/Tem': '#d62728',
 'GZMK+ Tem': '#aa40fc',
 'GZMK+ Tex': '#8c564b',
 'IFITM3+KLRG1+ Temra': '#e377c2',
 'ILTCK': '#b5bd61',
 'ITGAE+ Trm': '#17becf',
 'ITGB2+ Trm': '#aec7e8',
 'KLRG1+ Temra': '#ffbb78',
 'KLRG1- Temra': '#98df8a',
 'MAIT': '#ff9896',
 'SELL+ progenitor Tex': '#c49c94',
 'Tcm': '#f7b6d2',
 'Tn': '#dbdb8d',
    'None': '#F7F7F7'
}
disease_type_palette = {'AML': '#E64B35',
 'COVID-19': '#4DBBD5',
 'Healthy': '#029F87',
 'Inflammation': '#3C5488',
 'Inflammation-irAE': '#F39B7F',
  'CPI-irAE': '#F39B7F', 
 'Solid tumor': '#8491B4',
 'T-LGLL': '#91D1C2'}


# # Reading Data

# In[72]:


tcr_adata = sc.read_h5ad("./data/TCR.h5ad")
tcr_df = tcr_adata.obs


# In[71]:


# tcr_adata.write_h5ad("./data/TCR.h5ad")


# In[26]:


gex_adata = sc.read_h5ad("./data/GEX.h5ad")


# In[66]:


# gex_adata.write_h5ad("./data/GEX.h5ad")


# In[19]:


raw_gex_adata = sc.read_h5ad("./data/GEX.RAW.h5ad")


# In[213]:


tcr_adata.obs['tcr'] = list(map(lambda x: '-'.join(x), 
     tcr_adata.obs.loc[:,TRAB_DEFINITION + ['individual']].to_numpy()
))
tcr_adata.obs['tcr_no_individual'] = list(map(lambda x: '-'.join(x), 
     tcr_adata.obs.loc[:,TRAB_DEFINITION].to_numpy()
))


# In[214]:


gex_adata.obs['tcr'] = list(map(lambda x: '-'.join(x), 
     gex_adata.obs.loc[:,TRAB_DEFINITION_ORIG + ['individual']].to_numpy()
))
gex_adata.obs['tcr_no_individual'] = list(map(lambda x: '-'.join(x), 
     gex_adata.obs.loc[:,TRAB_DEFINITION_ORIG].to_numpy()
))


# # Basic Statistics

# ## Dataset statistics

# ### [Figure 1B]

# Pie chart illustrating the distribution of T cell lineages from different tissue origins and disease types in the collected data.

# In[47]:


plt.rcParams['font.size'] = 2
agg_gex = gex_adata.obs.groupby([
    "disease","meta_tissue_type"
]).agg({"reannotated_prediction_2":Counter,"disease_type":len})


palette = CELL_TYPE_C
palette["CD40LG"] = '#5BC8D9'
palette["Naive CD8"] = '#A9D55D'
palette["Naive CD4"] = '#3C3354'
palette["Cycling"] = '#A7A7A7'

fig,axes=plt.subplots(6,6)
axes=axes.flatten()
cm_dict = {
    "CD4": '#1982c4',
    "Treg": '#6a4c93',
    "CD8": '#ffca3a',
    "MAIT": '#2a9d8f',
    "Unpredictive": "#D7D7D7",
    "Undefined": "#D7D7D7",
    "Unknown": "#D7D7D7",
    "Others": '#8ac926'
}
a = 0
for (i,j),anno in zip(agg_gex.index, agg_gex['reannotated_prediction_2']):
    if type(anno) == float:
        continue
    anno = dict(sorted(anno.items(), key=lambda x: x[0]))
    custom_pie2(axes[a], anno, palette, radius=0.7, width=0.65,setp=False)
    axes[a].set_title('{}-{}-{}'.format(i,j,sum(anno.values())))
    a += 1
for ax in axes:
    ax.spines['right'].set_color('none')     
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')     
    ax.spines['left'].set_color('none')
    for line in ax.yaxis.get_ticklines():
        line.set_markersize(5)
        line.set_color("#585958")
        line.set_markeredgewidth(0.)
    for line in ax.xaxis.get_ticklines():
        line.set_markersize(5)
        line.set_markeredgewidth(0.)
        line.set_color("#585958")
    ax.set_xticklabels([])
    ax.set_yticklabels([])


# ## TCR Statistics

# In[37]:


print(
    "Total number of TCRs",
    len(tcr_df)
)
print(
    "Number of unique alpha chains", 
    len(np.unique(
        list(
            map(
                lambda x: '='.join(x), 
                tcr_df.loc[:,TRA_DEFINITION].to_numpy())
            )
        )
    )
)
print(
    "Number of unique beta chains", 
    len(np.unique(
        list(
            map(
                lambda x: '='.join(x), 
                tcr_df.loc[:,TRB_DEFINITION].to_numpy())
            )
        )
    )
)
print(
    "Number of unique alpha-beta chains", 
    len(np.unique(
        list(
            map(
                lambda x: '='.join(x), 
                tcr_df.loc[:,TRAB_DEFINITION].to_numpy())
            )
        )
    )
)


# ### [Figure S3E]

# In[49]:


from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 8

fig,axes=createSubplots(2,2)
axes=axes.flatten()
fig.set_size_inches(4,4)
axes=axes.flatten()
i = 0

for tx in ['CDR3a_length', 'CDR3b_length', 'CDR3a_mr_length', 'CDR3b_mr_length']:
    ax = axes[i]
    for ty in ['CD4', 'CD8', 'MAIT', 'Treg']:
        _tmp = tcr_df[tcr_df['reannotated_prediction_3'] == ty]
    
        # best fit of data
        (mu, sigma) = norm.fit(_tmp[tx])
        print(tx, ty, '{:1f} {:1f}'.format(mu, sigma))
        # the histogram of the data
        n, bins, patches = ax.hist(
            _tmp[tx], 
            60,  
            alpha=0,
            density = True
        )

        # add a 'best fit' line
        y = norm.pdf ( bins, mu, sigma)
        l = ax.plot(bins, y, linewidth=1, color=CELL_TYPE_C[ty])
        ax.set_title(tx)
    i += 1
for ax in axes: ax.set_ybound(0,0.3)
for ax in axes[:2]: ax.set_xbound(0,30)
for ax in axes[2:]: ax.set_xbound(0,10)


# ## GEX statistics

# ### [Figure 1C]

# Uniform manifold approximation and projection (UMAP) representation and cell lineage annotation for variational autoencoder (VAE)-based data integration for million-level scRNA-seq data. The panel in the bottom right corner shows a zoomed-in UMAP for cycling CD4 Tconv, CD8 T, and CD4 Treg cells.

# In[67]:


fig,ax=createFig()
fig.set_size_inches(5,5)
sc.pl.umap(gex_adata,ax=ax, color='cell_type_1', palette=reannotated_prediction_palette)


# In[68]:


fig,ax=createFig()
fig.set_size_inches(5,5)
sc.pl.umap(gex_adata,ax=ax, color='cell_type_2', palette=reannotated_prediction_palette)


# In[69]:


fig,ax=createFig()
fig.set_size_inches(5,5)
sc.pl.umap(gex_adata,ax=ax, color='cell_type_3', palette=reannotated_prediction_palette)


# In[71]:


fig,ax=createFig()
fig.set_size_inches(5,5)
sc.pl.umap(gex_adata,ax=ax, color='cell_type_4', palette=reannotation_palette)


# ### [Figure S2A]

# The UMAP projection of the integrated million-level T cells colored by study names (left panel), and bar plot for the distribution of different studies correponding to major T cell types including CD4 Tconv, CD4 Treg, CD4 Tn, CD8 Tn, and MAIT cells (right panel)

# In[72]:


fig,ax=createFig()
fig.set_size_inches(5,5)
sc.pl.umap(gex_adata,ax=ax, color='study_name', palette=study_name_palette)


# ### [Figure 1D]

# Feature plot for key marker genes, including CD8A, CD4, FOXP3, TCF7, HAVCR2, GZMB, MKI67, CD40LG, KLRB1, ZBTB16, PDCD1, and CX3CR1, used for cell lineage annotation.

# In[21]:


# Normalize raw data for plotting gene expression
sc.pp.normalize_total(raw_gex_adata)
sc.pp.log1p(raw_gex_adata)

fig,ax=createFig()
fig.set_size_inches(5,5)
ax.scatter(
    raw_gex_adata.obsm["X_umap"][:,0],
    raw_gex_adata.obsm["X_umap"][:,1],
    s=0.1,
    linewidths=0,
    color='#FDF6F3'
)
gene_name = 'HAVCR2'
sc.pl.umap(
    raw_gex_adata[raw_gex_adata.X[:,list(raw_gex_adata.var.index).index(gene_name)].toarray() > 0], ax=ax, 
    color=gene_name, 
    cmap='Reds',
    s=0.5
)


# ### [Figure S2C]

# Cell type annotation by singleR as previously used in Wu et al., 2021.

# In[73]:


fig,ax=createFig()
fig.set_size_inches(5,5)
sc.pl.umap(gex_adata, ax=ax, color='predictions')


# ### Figure S2B

# Distribution of cells from different disease states including solid tumors, inflammation, CPI-irAE (checkpoint inhibitor associated immune-related adverse events), AML (Acute Myeloid Leukemia), COVID-19, Healthy, and T-LGLL (T cell large granular lymphocytic leukemia) in the integrated UMAP and pie chart for the composition of T cell states in each disease conditions.

# In[75]:


fig,ax=createFig()
fig.set_size_inches(5,5)
ax.scatter(
    gex_adata.obsm["X_umap"][:,0],
    gex_adata.obsm["X_umap"][:,1],
    s=0.1,
    linewidths=0,
    color='#F7F7F7'
)
sc.pl.umap(
    gex_adata[gex_adata.obs['disease_type'] == 'Solid tumor'], 
    ax=ax, 
    color='disease_type', 
    s=0.5,
    palette=disease_type_palette
)


# ### Figure 1E

# Pie chart depicting the composition of cell lineages for all TCRα/β pairs (upper panel) and unique TCRα/β pairs (lower panel).

# In[109]:


anno = Counter(gex_adata.obs['cell_type_3'])
print(anno)
fig,axes=plt.subplots(1,2)
custom_pie2(axes[0],anno,reannotated_prediction_palette)
axes[0].set_title("GEX")
anno = Counter(tcr_df['cell_type_3'])
print(anno)
custom_pie2(axes[1],anno,reannotated_prediction_palette)
axes[1].set_title("TCR")
plt.savefig("./figures/gex_tcr_df_cell_type_3_pie.pdf")


# # VJ-segment joining analysis

# ## Single VJ segment and T cell type

# In[114]:


import matplotlib.lines as mlines
from scipy import stats
import warnings
plt.rcParams['font.size'] = 8

warnings.filterwarnings("ignore")


# Func to draw line segment
result = []
for cell_type in np.unique(tcr_df['cell_type_3']):
    for gene in [
        'TRAV','TRAJ','TRBV',"TRBJ"
    ]:

        c = Counter(list(map(lambda x: x, tcr_df.loc[
            tcr_df['cell_type_3'] == cell_type,
            gene
        ].to_numpy())))
        c_non = Counter(list(map(lambda x: x, tcr_df.loc[
            tcr_df['cell_type_3'] != cell_type,
            gene
        ].to_numpy())))

        for i in c.keys():
            mat = np.array([
                [c[i], sum(c.values()) - c[i]],
                [c_non[i], sum(c_non.values()) - c_non[i]]
            ])
            odds_ratio, pvalue = stats.fisher_exact(mat)
            chi, chi_p, _, _  = stats.chi2_contingency(mat)
            lower95ci = np.exp(np.log(odds_ratio) - 1.96 * np.sqrt(sum(list(map(lambda x: 1/x, mat.flatten())))))
            upper95ci = np.exp(np.log(odds_ratio) + 1.96 * np.sqrt(sum(list(map(lambda x: 1/x, mat.flatten())))))
            result.append([cell_type,
                           i, 
                           odds_ratio, 
                           lower95ci, 
                           upper95ci, 
                           pvalue,
                           chi_p,
                           c[i], 
                           c_non[i], 
                           sum(c.values()), 
                           sum(c_non.values()) 
                          ])


# In[115]:


from statsmodels.stats.multitest import multipletests
single_vj_cell_type = pd.DataFrame(result, columns=['cell_type','segment','odds','lower95ci','upper95ci','pvalue','chi_pvalue','count_pos','count_neg','sum_pos','sum_neg'])
single_vj_cell_type = single_vj_cell_type[single_vj_cell_type['cell_type'] != 'Ambiguous']
single_vj_cell_type['pvalue'] = single_vj_cell_type['pvalue'] + 1e-320
single_vj_cell_type['chi_pvalue'] = single_vj_cell_type['chi_pvalue'] + 1e-320
single_vj_cell_type['adj_pvalue'] = multipletests(np.array(single_vj_cell_type['pvalue']))[1]
single_vj_cell_type['adj_chi_pvalue'] = multipletests(np.array(single_vj_cell_type['chi_pvalue']))[1]


# In[116]:


single_vj_cell_type.to_csv("./data/single_vj_cell_type.csv", index=False)


# In[117]:


import warnings
warnings.filterwarnings('ignore')
fig,axes=createSubplots(2,2)
axes=axes.flatten()
for ax,cell_type in zip(axes,['CD8','CD4','MAIT','Treg']):
    NormalizeData = lambda data: (data - np.min(data)) / (np.max(data) - np.min(data))
    ax.scatter(np.log2(single_vj_cell_type[single_vj_cell_type['cell_type'] == cell_type]['odds']), 
               -np.log10(single_vj_cell_type[single_vj_cell_type['cell_type'] == cell_type]['adj_pvalue']), 
               s=NormalizeData(single_vj_cell_type[single_vj_cell_type['cell_type'] == cell_type]['count_pos']) * 10, 
               c=list(map(lambda x: CELL_TYPE_C[x], single_vj_cell_type[single_vj_cell_type['cell_type'] == cell_type]['cell_type'])))
    ax.grid(alpha=0.2, linewidth = 0.5)
    for x,y,s,i in zip(
        np.log2(single_vj_cell_type[single_vj_cell_type['cell_type'] == cell_type]['odds']), 
        -np.log10(single_vj_cell_type[single_vj_cell_type['cell_type'] == cell_type]['adj_pvalue']),
        list(single_vj_cell_type[single_vj_cell_type['cell_type'] == cell_type]['segment']),
        list(single_vj_cell_type[single_vj_cell_type['cell_type'] == cell_type]['cell_type'])
    ):
        if x > 0.5 and y > 50:
            ax.text(x+0.1,y,s,c=CELL_TYPE_C[i])
    ax.set_title(cell_type)
    ax.set_xlabel("Log2(FoldChange)")
    ax.set_ylabel("-Log10(pVal)")


# ## Paired VJ segment and T cell type

# In[121]:


import matplotlib.lines as mlines
from scipy import stats
from itertools import combinations

plt.rcParams['font.size'] = 8
# Func to draw line segment
result = []
for i,j in combinations(['TRAV','TRAJ','TRBV',"TRBJ"],2):
    tcr_df[f'{i}_{j}'] = list(map(lambda x: '='.join(x), zip(tcr_df[i],tcr_df[j])))

for cell_type in np.unique(tcr_df['cell_type_3']):
    for gene in list(map('_'.join, combinations(['TRAV','TRAJ','TRBV',"TRBJ"],2))):
        c = Counter(list(map(lambda x: x, tcr_df.loc[
            tcr_df['cell_type_3'] == cell_type,
            gene
        ].to_numpy())))
        c_non = Counter(list(map(lambda x: x, tcr_df.loc[
            tcr_df['cell_type_3'] != cell_type,
            gene
        ].to_numpy())))


        for i in c.keys():
            mat = np.array([
                [c[i], sum(c.values()) - c[i]],
                [c_non[i], sum(c_non.values()) - c_non[i]]
            ])
            odds_ratio, pvalue = stats.fisher_exact(mat)
            chi, chi_p, _, _  = stats.chi2_contingency(mat)
            lower95ci = np.exp(np.log(odds_ratio) - 1.96 * np.sqrt(sum(list(map(lambda x: 1/x, mat.flatten())))))
            upper95ci = np.exp(np.log(odds_ratio) + 1.96 * np.sqrt(sum(list(map(lambda x: 1/x, mat.flatten())))))
            result.append([cell_type,
                           i, 
                           odds_ratio, 
                           lower95ci, 
                           upper95ci, 
                           pvalue, 
                           chi_p,
                           c[i], 
                           c_non[i], 
                           sum(c.values()), 
                           sum(c_non.values()) 
                          ])


# In[122]:


dual_vj_cell_type = pd.DataFrame(result, columns=['cell_type','segment','odds','lower95ci','upper95ci','pvalue','chi_pvalue','count_pos','count_neg','sum_pos','sum_neg'])
dual_vj_cell_type = dual_vj_cell_type[dual_vj_cell_type['cell_type'] != 'Ambiguous']
dual_vj_cell_type['pvalue'] = dual_vj_cell_type['pvalue'] + 1e-320
dual_vj_cell_type['adj_pvalue'] = multipletests(np.array(dual_vj_cell_type['pvalue']))[1]


# In[123]:


import warnings
warnings.filterwarnings('ignore')
fig,axes=createSubplots(2,2)
axes=axes.flatten()
for ax,cell_type in zip(axes,['CD8','CD4','MAIT','Treg']):
    NormalizeData = lambda data: (data - np.min(data)) / (np.max(data) - np.min(data))
    ax.scatter(np.log2(dual_vj_cell_type[dual_vj_cell_type['cell_type'] == cell_type]['odds']), 
               -np.log10(dual_vj_cell_type[dual_vj_cell_type['cell_type'] == cell_type]['adj_pvalue']), 
               s=NormalizeData(dual_vj_cell_type[dual_vj_cell_type['cell_type'] == cell_type]['count_pos']) * 10, 
               c=list(map(lambda x: CELL_TYPE_C[x], dual_vj_cell_type[dual_vj_cell_type['cell_type'] == cell_type]['cell_type'])))
    ax.grid(alpha=0.2, linewidth = 0.5)
    for x,y,s,i in zip(
        np.log2(dual_vj_cell_type[dual_vj_cell_type['cell_type'] == cell_type]['odds']), 
        -np.log10(dual_vj_cell_type[dual_vj_cell_type['cell_type'] == cell_type]['adj_pvalue']),
        list(dual_vj_cell_type[dual_vj_cell_type['cell_type'] == cell_type]['segment']),
        list(dual_vj_cell_type[dual_vj_cell_type['cell_type'] == cell_type]['cell_type'])
    ):
        if x > 0.5 and y > 50:
            continue
            ax.text(x+0.1,y,s,c=CELL_TYPE_C[i])
    ax.set_title(cell_type)
    ax.set_xlabel("Log2(FoldChange)")
    ax.set_ylabel("-Log10(pVal)")


# ### [Figure2A]

# Sankey plot illustrating the enriched TRAV/TRAJ, TRAV/TRBV, and TRBV/TRBJ pairing and joining preferences in CD8 T, MAIT, and CD4 Tconv cells. The figure is generated by D3Js.

# In[125]:


_df = dual_vj_cell_type.loc[
    np.array(dual_vj_cell_type['adj_pvalue'] < 0.05) & 
    np.array(dual_vj_cell_type['odds'] > 2) & 
    np.array(dual_vj_cell_type['count_pos'] > 350) & 
    np.array(list(map(lambda x: (x.startswith("TRAV") and x.split("=")[1].startswith("TRAJ")) or 
                                (x.startswith("TRAV") and x.split("=")[1].startswith("TRBV")) or 
                                (x.startswith("TRBV") and x.split("=")[1].startswith("TRBJ")), 
    dual_vj_cell_type['segment'])))
].loc[:,['cell_type','segment','odds','count_pos']]



_df = pd.DataFrame(
    list(map(lambda x: [x[0],
                        x[1].split("=")[0], 
                        x[1].split("=")[1],
                        x[2],
                        x[3]] if not x[1].split("=")[1].startswith('TRAJ') else \
                       [x[0],
                        x[1].split("=")[1], 
                        x[1].split("=")[0],
                        x[2],
                        x[3]], _df.to_numpy())),
    columns=['preference','source','target','odds','value']
)
_df.to_csv("./sankey/files/cell_type_specific_vj_pairing.csv", index=False)


# In[24]:


from IPython.core.display import HTML
HTML(''.join(open("./sankey/saved.html").readlines()))


# ### [Figure 2B]

# In[38]:


fig,ax=createFig()
ax.spines['left'].set_color('none')     
ax.spines['bottom'].set_color('none')

_df = pd.read_csv("./sankey/files/cell_type_specific_vj_pairing.csv")

gex_adata.obs['VJ_usage_group']  = 'NA'
for i in _df[_df['preference'] == 'CD4'].loc[:,['source','target']].to_numpy():
    if i[0].startswith('TRAV') and i[1].startswith("TRBV"):
        gex_adata.obs.loc[
            np.array(gex_adata.obs['IR_VJ_1_v_call'] == i[0]) &
            np.array(gex_adata.obs['IR_VDJ_1_v_call'] == i[1]),
            'VJ_usage_group'
        ] = i[0] + i[1]

fig.set_size_inches(5,5)
obsm = gex_adata[list(map(lambda x: x == 'NA', gex_adata.obs['VJ_usage_group']))].obsm
ax.scatter(obsm["X_umap"][:,0], obsm["X_umap"][:,1], s=0.01, color="#F1F1F1")
obsm = gex_adata[list(map(lambda x: x != 'NA', gex_adata.obs['VJ_usage_group']))].obsm['X_umap']
sns.kdeplot(
    x= obsm[:,0],
    y= obsm[:,1],
    cmap=make_colormap(['#F1F1F1',CELL_TYPE_C["CD4"]]), shade=True, bw_adjust=0.4,
    ax=ax
)
ax.set_xticks([])
ax.set_yticks([])


# ### [Figure 2C]

# In[141]:


raw_gex_adata.obs['VJPairing'] = '-'
vj_map = dict(zip(TRAB_DEFINITION, TRAB_DEFINITION_ORIG))
for i,j in _df.loc[:,['source','target']].to_numpy():
    if i.startswith("TRAV") and j.startswith("TRBV"):
        raw_gex_adata.obs.loc[
            np.array(raw_gex_adata.obs[vj_map[i[:4]]] == i) & 
            np.array(raw_gex_adata.obs[vj_map[j[:4]]] == j),'VJPairing'
        ] = f'{i}-{j}'

sc.pl.dotplot(
    raw_gex_adata, 
    ['CD4','CD8A','CD8B','ZBTB16','KLRB1','SLC4A10'], 
    groupby='VJPairing', 
    standard_scale='var',
    swap_axes=True,
    save='vj_pairing_cell_type_specific'
)


# # CDR region and T cell type

# In[ ]:


CDR3mr = tcr_df.loc[:,
    ['CDR3a','CDR3a_mr','CDR3b_mr','cell_type_3','individual','CDR3a_length','CDR3b_length','CDR1a','CDR1b','CDR2a','CDR2b'] + TRAB_DEFINITION
]


# In[ ]:


for k in ["TRAV",'TRAJ','TRBV','TRBJ']:
    for i in np.unique(CDR3mr.loc[:,k].to_numpy().flatten()):
        CDR3mr[i] = list(map(lambda x: 1 if x else 0, CDR3mr[k] == i))
    CDR3mr.pop(k)


# In[ ]:


properties = list(np.unique(list(AMINO_ACID_PROPERTIES.values())))
mat = np.zeros((len(CDR3mr), len(properties)))
for idx,i in zip(CDR3mr.index, CDR3mr['CDR3a_mr'].to_numpy()):
    if type(i) != str :
        continue
    i = list(map(AMINO_ACID_PROPERTIES.get, list(i)))
    l = len(i)
    counter = {k:v / l for k,v in Counter(i).items()}
    for k,v in counter.items():
        mat[idx,properties.index(k)] = v * 100
CDR3mr = CDR3mr.join(
    pd.DataFrame(mat,index=CDR3mr.index, columns=list(map(lambda x:'CDR3a_mr_' + x, properties)))
)
mat = np.zeros((len(CDR3mr), len(properties)))
for idx,i in zip(CDR3mr.index, CDR3mr['CDR3b_mr'].to_numpy()):
    if type(i) != str :
        continue
    l = len(i)
    i = list(map(AMINO_ACID_PROPERTIES.get, list(i)))
    counter = {k:v / l for k,v in Counter(i).items()}
    for k,v in counter.items():
        mat[idx,properties.index(k)] = v * 100
CDR3mr = CDR3mr.join(
    pd.DataFrame(mat,index=CDR3mr.index, columns=list(map(lambda x:'CDR3b_mr_' + x, properties)))
)


# In[ ]:


mat = np.zeros((len(CDR3mr), len(AMINO_ACIDS)))
for idx,i in zip(CDR3mr.index, CDR3mr['CDR3a_mr'].to_numpy()):
    if type(i) != str :
        continue
    l = len(i)
    counter = {k:v / l for k,v in Counter(i).items()}
    for k,v in counter.items():
        mat[idx,AMINO_ACIDS.index(k)] = v * 100
CDR3mr = CDR3mr.join(
    pd.DataFrame(mat,index=CDR3mr.index, columns=list(map(lambda x:'CDR3a_mr_' + x, AMINO_ACIDS)))
)
mat = np.zeros((len(CDR3mr), len(AMINO_ACIDS)))
for idx,i in zip(CDR3mr.index, CDR3mr['CDR3b_mr'].to_numpy()):
    if type(i) != str :
        continue
    l = len(i)
    counter = {k:v / l for k,v in Counter(i).items()}
    for k,v in counter.items():
        mat[idx,AMINO_ACIDS.index(k)] = v * 100
CDR3mr = CDR3mr.join(
    pd.DataFrame(mat,index=CDR3mr.index, columns=list(map(lambda x:'CDR3b_mr_' + x, AMINO_ACIDS)))
)


# In[ ]:


properties = list(np.unique(list(AMINO_ACID_PROPERTIES.values())))
mat = np.zeros((len(CDR3mr), len(properties)))
for kk in ['CDR1a','CDR1b','CDR2a','CDR2b']:
    for idx,i in zip(CDR3mr.index, CDR3mr[kk].to_numpy()):
        if type(i) != str :
            continue
        i = list(map(AMINO_ACID_PROPERTIES.get, list(i)))
        l = len(i)
        counter = {k:v / l for k,v in Counter(i).items()}
        for k,v in counter.items():
            mat[idx,properties.index(k)] = v * 100
    CDR3mr = CDR3mr.join(
        pd.DataFrame(mat,index=CDR3mr.index, columns=list(map(lambda x:f'{kk}_' + x, properties)))
    )


# In[ ]:


mat = np.zeros((len(CDR3mr), len(AMINO_ACIDS)))
for kk in ['CDR1a','CDR1b','CDR2a','CDR2b']:
    for idx,i in zip(CDR3mr.index, CDR3mr[kk].to_numpy()):
        if type(i) != str :
            continue
        l = len(i)
        counter = {k:v / l for k,v in Counter(i).items()}
        for k,v in counter.items():
            mat[idx,AMINO_ACIDS.index(k)] = v * 100
    CDR3mr = CDR3mr.join(
        pd.DataFrame(mat,index=CDR3mr.index, columns=list(map(lambda x:kk + "_" + x, AMINO_ACIDS)))
    )


# In[150]:


CDR3mr.to_csv("./data/cdr3mr_analysis.csv", index=False)


# In[143]:


CDR3mr = pd.read_csv("./data/cdr3mr_analysis.csv")


# In[175]:


from scipy.stats import ttest_ind, ttest_rel
_agg = tcr_df.groupby("individual").agg({"cell_type_3": lambda x: dict(Counter(x))})
celltype_by_individual = dict(zip(_agg.index, FLATTEN(list(_agg.to_numpy()))))
_celltype_by_individual = {k:{} for k in celltype_by_individual.keys()}
for i,j in celltype_by_individual.items():
    for k,v in j.items():
        if k == 'Treg':
            _celltype_by_individual[i]["1_Treg"] = v
        if k == 'CD4':
            _celltype_by_individual[i]["2_CD4"] = v
        if k == 'CD8':
            _celltype_by_individual[i]["3_CD8"] = v
        else:
            _celltype_by_individual[i][k] = v


# ### [Figure 2D]

# In[177]:


plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=None)

cdr3_amino_acid_properties = []
properties = list(np.unique(list(AMINO_ACID_PROPERTIES.values())))
for c in ['CDR1a','CDR1b','CDR2a','CDR2b','CDR3a_mr','CDR3b_mr']:
    fig,axes=createSubplots(1,5)
    fig.set_size_inches(10,3)
    for ax, k in zip(axes, list(map(lambda x: f'{c}_' + x, properties))):
        agg = CDR3mr.groupby(["cell_type_3","individual"]).agg({k:np.mean})
        agg = pandas_aggregation_to_wide(agg)
        agg['cell_type_3'] = agg['cell_type_3'].replace("Treg",'1_Treg').replace("CD4",'2_CD4').replace("CD8","3_CD8")
        agg = agg.sort_values('cell_type_3', ascending=True)
        
        agg['count'] = list(map(lambda x: _celltype_by_individual[x[0]][x[1]], zip(agg['individual'],agg['cell_type_3'])))
        agg = agg[agg['count'] > 100]
        agg = agg[list(map(lambda x: x in ['1_Treg','2_CD4','3_CD8'], agg['cell_type_3']))]

        sns.violinplot(data=agg, x='cell_type_3',y=k, showfliers=False,ax=ax,palette={'1_Treg':'#6B4E94','2_CD4':'#1E83C5','3_CD8':'#FFCB39'})
        sns.stripplot(data=agg, x='cell_type_3',y=k, color='black', alpha=0.5, linewidth=0, size=2, jitter=False, ax=ax)
        for i in agg[agg['cell_type_3'] != '3_CD8'].groupby("individual").agg({k:list}).to_numpy():
            if len(i[0]) == 2:
                ax.plot((0,1),i[0],c='gray',alpha=0.5,lw=0.3)
        for i in agg[agg['cell_type_3'] != '1_Treg'].groupby("individual").agg({k:list}).to_numpy():
            if len(i[0]) == 2:
                ax.plot((1,2),i[0],c='gray',alpha=0.5,lw=0.3)  
        treg,cd4,cd8=list(map(list, agg.groupby("cell_type_3").agg({k:list}).to_numpy()))
        agg_celltype = agg.groupby(["cell_type_3","individual"]).agg({k:list})
        common_indices = list(set(agg_celltype.loc['1_Treg'].index).intersection(set(agg_celltype.loc['2_CD4'].index)))
        treg, cd4 = (
            list(agg_celltype.loc[list(map(lambda x: ('1_Treg',x ), common_indices))][k]),
            list(agg_celltype.loc[list(map(lambda x: ('2_CD4',x ), common_indices))][k])
        )
        stat = ttest_rel(treg,cd4)
        cdr3_amino_acid_properties.append((k, 'Treg vs CD4', np.mean(treg), np.mean(cd4), stat.pvalue[0]))
        
        common_indices = list(set(agg_celltype.loc['3_CD8'].index).intersection(set(agg_celltype.loc['2_CD4'].index)))
        cd4,cd8 = (
            list(agg_celltype.loc[list(map(lambda x: ('2_CD4',x ), common_indices))][k]),
            list(agg_celltype.loc[list(map(lambda x: ('3_CD8',x ), common_indices))][k])
        )
        stat = ttest_rel(cd4,cd8)
        cdr3_amino_acid_properties.append((k, 'CD4 vs CD8', np.mean(cd4), np.mean(cd8), stat.pvalue[0]))
        
        common_indices = list(set(agg_celltype.loc['1_Treg'].index).intersection(set(agg_celltype.loc['3_CD8'].index)))
        treg,cd8 = (
            list(agg_celltype.loc[list(map(lambda x: ('1_Treg',x ), common_indices))][k]),
            list(agg_celltype.loc[list(map(lambda x: ('3_CD8',x ), common_indices))][k])
        )
        stat = ttest_rel(treg,cd8)
        cdr3_amino_acid_properties.append((k, 'Treg vs CD8', np.mean(treg), np.mean(cd8), stat.pvalue[0]))
        
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(k.split("_")[-1])
    fig.savefig(f"./figures//huardb_{c}_CD4vsCD8vsTreg_grouped_properties.pdf")
    
    


# In[179]:


# pd.DataFrame(cdr3_amino_acid_properties).to_csv("./data/20230307_cdr_amino_acid_properties.csv",index=False,header=None)
cdr3_amino_acid_properties = pd.DataFrame(cdr3_amino_acid_properties)


# ### [Figure S3A-D, F]

# In[184]:


cdr_amino_acid_properties = []

for c in ['CDR1a','CDR1b','CDR2a','CDR2b','CDR3a_mr','CDR3b_mr']:
    fig,axes=createSubplots(1,len(AMINO_ACIDS))
    fig.set_size_inches(42,3)
    for ax, k in zip(axes, list(map(lambda x: f'{c}_' + x, AMINO_ACIDS))):
        agg = CDR3mr.groupby(["cell_type_3","individual"]).agg({k:np.mean})
        agg = pandas_aggregation_to_wide(agg)
        agg['cell_type_3'] = agg['cell_type_3'].replace("Treg",'1_Treg').replace("CD4",'2_CD4').replace("CD8","3_CD8")
        agg = agg.sort_values('cell_type_3', ascending=True)
        
        agg['count'] = list(map(lambda x: _celltype_by_individual[x[0]][x[1]], zip(agg['individual'],agg['cell_type_3'])))
        agg = agg[agg['count'] > 100]
        agg = agg[list(map(lambda x: x in ['1_Treg','2_CD4','3_CD8'], agg['cell_type_3']))]

        sns.violinplot(data=agg, x='cell_type_3',y=k, showfliers=False,ax=ax,palette={'1_Treg':'#6B4E94','2_CD4':'#1E83C5','3_CD8':'#FFCB39'})
        sns.stripplot(data=agg, x='cell_type_3',y=k, color='black', alpha=0.5, linewidth=0, size=2, jitter=False, ax=ax)
        for i in agg[agg['cell_type_3'] != '3_CD8'].groupby("individual").agg({k:list}).to_numpy():
            if len(i[0]) == 2:
                ax.plot((0,1),i[0],c='gray',alpha=0.5,lw=0.3)
        for i in agg[agg['cell_type_3'] != '1_Treg'].groupby("individual").agg({k:list}).to_numpy():
            if len(i[0]) == 2:
                ax.plot((1,2),i[0],c='gray',alpha=0.5,lw=0.3)  
        agg_celltype = agg.groupby(["cell_type_3","individual"]).agg({k:list})
        common_indices = list(set(agg_celltype.loc['1_Treg'].index).intersection(set(agg_celltype.loc['2_CD4'].index)))
        treg, cd4 = (
            list(agg_celltype.loc[list(map(lambda x: ('1_Treg',x ), common_indices))][k]),
            list(agg_celltype.loc[list(map(lambda x: ('2_CD4',x ), common_indices))][k])
        )
        stat = ttest_rel(treg,cd4)
        cdr_amino_acid_properties.append((k, 'Treg vs CD4', np.mean(treg), np.mean(cd4), stat.pvalue[0]))
        
        common_indices = list(set(agg_celltype.loc['3_CD8'].index).intersection(set(agg_celltype.loc['2_CD4'].index)))
        cd4,cd8 = (
            list(agg_celltype.loc[list(map(lambda x: ('2_CD4',x ), common_indices))][k]),
            list(agg_celltype.loc[list(map(lambda x: ('3_CD8',x ), common_indices))][k])
        )
        stat = ttest_rel(cd4,cd8)
        cdr_amino_acid_properties.append((k, 'CD4 vs CD8', np.mean(cd4), np.mean(cd8), stat.pvalue[0]))
        
        common_indices = list(set(agg_celltype.loc['1_Treg'].index).intersection(set(agg_celltype.loc['3_CD8'].index)))
        treg,cd8 = (
            list(agg_celltype.loc[list(map(lambda x: ('1_Treg',x ), common_indices))][k]),
            list(agg_celltype.loc[list(map(lambda x: ('3_CD8',x ), common_indices))][k])
        )
        stat = ttest_rel(treg,cd8)
        cdr_amino_acid_properties.append((k, 'Treg vs CD8', np.mean(treg), np.mean(cd8), stat.pvalue[0]))
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(k.split("_")[-1])
    fig.savefig(f"./figures/huardb_{c}_CD4vsCD8vsTreg_AminoAcids.pdf")
    


# # Recurrent TCR analysis

# ## Recurrent TCR (alpha-beta)

# In[185]:


count = {}
for k,i in zip(list(map('='.join, tcr_df.loc[:,[
    'CDR3a','CDR3b','TRAV','TRAJ','TRBV','TRBJ'
]].to_numpy())), tcr_df['individual']):

    if k in count.keys():
        if i not in count[k]: count[k].append(i)
    else:
        count[k] = [i]

recurrent_tcrs_alpha_beta = pd.DataFrame(
    list(map(
        lambda x: x[0].split("="), 
        list(filter(lambda x: len(x[1]) > 1, count.items()))
    )),
    columns=["CDR3a","CDR3b","TRAV","TRAJ","TRBV","TRBJ"]
)

recurrent_tcrs_alpha_beta['individuals'] = list(map(
        lambda x: ','.join(x[1]),
        list(filter(lambda x: len(x[1]) > 1, count.items()))
    ))
recurrent_tcrs_alpha_beta['number_of_individuals'] = list(map(
        lambda x:len(x[1]),
        list(filter(lambda x: len(x[1]) > 1, count.items()))
    ))
recurrent_tcrs_alpha_beta['disease'] = list(map(lambda x: 
     ','.join(
         list(np.unique(list(filter(lambda a: a, 
                             map(lambda z:individual2disease_type.get(z), x.split(",")))
     )))
     ), recurrent_tcrs_alpha_beta['individuals'])
)
tcr_df['recurrent_tcrs_alpha_beta'] = '-'
tcr_df['recurrent_tcrs_alpha_beta_n_individuals'] = 0
m = dict(zip(list(map(
    '='.join, 
    recurrent_tcrs_alpha_beta.loc[:,['CDR3a','CDR3b','TRAV','TRAJ','TRBV','TRBJ']].to_numpy())),
    recurrent_tcrs_alpha_beta['individuals']
))
n = dict(zip(list(map(
    '='.join, 
    recurrent_tcrs_alpha_beta.loc[:,['CDR3a','CDR3b','TRAV','TRAJ','TRBV','TRBJ']].to_numpy())),
    recurrent_tcrs_alpha_beta['number_of_individuals']
))
_tmp_1 = []
_tmp_2 = []
pbar = tqdm.tqdm(total=len(tcr_df))
for i,k in enumerate(
    list(map('='.join, tcr_df.loc[:,[
    'CDR3a','CDR3b','TRAV','TRAJ','TRBV','TRBJ'
    ]].to_numpy()))
):
    if k in m.keys():
        _tmp_1.append(m[k])
        _tmp_2.append(n[k])
        assert(len(m[k].split(',')) == n[k])
    else:
        _tmp_1.append('-')
        _tmp_2.append(0)
    pbar.update(1)
tcr_df['recurrent_tcrs_alpha_beta'] = _tmp_1
tcr_df['recurrent_tcrs_alpha_beta_n_individuals'] = _tmp_2
pbar.close()


# ## Recurrent TCR (beta)

# In[188]:


count = {}
for k,i in zip(list(map('='.join, tcr_df.loc[:,[
    'CDR3b','TRBV','TRBJ'
]].to_numpy())), tcr_df['individual']):
    if k in count.keys() and i not in count[k]:
        count[k].append(i)
    else:
        count[k] = [i]
recurrent_tcrs_beta = pd.DataFrame(
    list(map(
        lambda x: x[0].split("="), 
        list(filter(lambda x: len(x[1]) > 1, count.items()))
    )),
    columns=["CDR3b","TRBV","TRBJ"]
)

recurrent_tcrs_beta['individuals'] = list(map(
        lambda x: ','.join(x[1]),
        list(filter(lambda x: len(x[1]) > 1, count.items()))
    ))
recurrent_tcrs_beta['number_of_individuals'] = list(map(
        lambda x:len(x[1]),
        list(filter(lambda x: len(x[1]) > 1, count.items()))
    ))
tcr_df['recurrent_tcrs_beta'] = '-'
tcr_df['recurrent_tcrs_beta_n_individuals'] = 0
m = dict(zip(list(map(
    '='.join, 
    recurrent_tcrs_beta.loc[:,['CDR3b','TRBV',"TRBJ"]].to_numpy())),
    recurrent_tcrs_beta['individuals']
))
n = dict(zip(list(map(
    '='.join, 
    recurrent_tcrs_beta.loc[:,['CDR3b','TRBV',"TRBJ"]].to_numpy())),
    recurrent_tcrs_beta['number_of_individuals']
))

_tmp_1 = []
_tmp_2 = []

pbar = tqdm.tqdm(total=len(tcr_df))
for i,k in enumerate(
    list(map('='.join, tcr_df.loc[:,[
    'CDR3b','TRBV','TRBJ',
    ]].to_numpy()))
):
    if k in m.keys():
        _tmp_1.append(m[k])
        _tmp_2.append(n[k])
    else:
        _tmp_1.append('-')
        _tmp_2.append(0)
    pbar.update(1)
tcr_df['recurrent_tcrs_beta'] = _tmp_1
tcr_df['recurrent_tcrs_beta_n_individuals'] = _tmp_2
pbar.close()


# ## Recurrent TCR (alpha)

# In[189]:


count = {}
for k,i in zip(list(map('='.join, tcr_df.loc[:,[
    'CDR3a','TRAV','TRAJ'
]].to_numpy())), tcr_df['individual']):
    if k in count.keys() and i not in count[k]:
        count[k].append(i)
    else:
        count[k] = [i]
recurrent_tcrs_alpha = pd.DataFrame(
    list(map(
        lambda x: x[0].split("="), 
        list(filter(lambda x: len(x[1]) > 1, count.items()))
    )),
    columns=["CDR3a","TRAV","TRAJ"]
)

recurrent_tcrs_alpha['individuals'] = list(map(
        lambda x: ','.join(x[1]),
        list(filter(lambda x: len(x[1]) > 1, count.items()))
    ))
recurrent_tcrs_alpha['number_of_individuals'] = list(map(
        lambda x:len(x[1]),
        list(filter(lambda x: len(x[1]) > 1, count.items()))
    ))
tcr_df['recurrent_tcrs_alpha'] = '-'
tcr_df['recurrent_tcrs_alpha_n_individuals'] = 0
m = dict(zip(list(map(
    '='.join, 
    recurrent_tcrs_alpha.loc[:,['CDR3a','TRAV',"TRAJ"]].to_numpy())),
    recurrent_tcrs_alpha['individuals']
))
n = dict(zip(list(map(
    '='.join, 
    recurrent_tcrs_alpha.loc[:,['CDR3a','TRAV',"TRAJ"]].to_numpy())),
    recurrent_tcrs_alpha['number_of_individuals']
))
_tmp_1 = []
_tmp_2 = []
pbar = tqdm.tqdm(total=len(tcr_df))
for i,k in enumerate(
    list(map('='.join, tcr_df.loc[:,[
    'CDR3a','TRAV','TRAJ',
    ]].to_numpy()))
):
    if k in m.keys():
        _tmp_1.append(m[k])
        _tmp_2.append(n[k])
    else:
        _tmp_1.append('-')
        _tmp_2.append(0)
    pbar.update(1)
tcr_df['recurrent_tcrs_alpha'] = _tmp_1
tcr_df['recurrent_tcrs_alpha_n_individuals'] = _tmp_2
pbar.close()


# ## Summary of recurrent TCR

# In[28]:


tcr_df['recurrent_tcrs_alpha_beta_disease'] = list(map(lambda x: ','.join(
         list(np.unique(list(filter(lambda a: a, 
         map(lambda z:individual2disease_type.get(z.strip(),'Unknown'), x.split(",")))
     )))) if x != '-' and type(x) == str else x, tcr_df['recurrent_tcrs_alpha_beta']))

tcr_df['recurrent_tcrs_alpha_disease'] = list(map(lambda x:     ','.join(
         list(np.unique(list(filter(lambda a: a, 
         map(lambda z:individual2disease_type.get(z.strip(),'Unknown'), x.split(",")))
     )))) if x != '-' and type(x) == str else x, tcr_df['recurrent_tcrs_alpha']))

tcr_df['recurrent_tcrs_beta_disease'] = list(map(lambda x:     ','.join(
         list(np.unique(list(filter(lambda a: a, 
         map(lambda z:individual2disease_type.get(z.strip(),'Unknown'), x.split(",")))
     )))) if x != '-' and type(x) == str else x, tcr_df['recurrent_tcrs_beta']))
print("Number of recurrent Alpha Beta TCR", len(np.unique(list(map('='.join, tcr_df[
    tcr_df['recurrent_tcrs_alpha_beta_disease'] != '-'
].loc[:,TRAB_DEFINITION].to_numpy())))))

print("Number of recurrent Beta TCR", len(np.unique(list(map('='.join, tcr_df[
    tcr_df['recurrent_tcrs_beta_disease'] != '-'
].loc[:,TRB_DEFINITION].to_numpy())))))

print("Number of recurrent Alpha TCR", len(np.unique(list(map('='.join, tcr_df[
    tcr_df['recurrent_tcrs_alpha_disease'] != '-'
].loc[:,TRA_DEFINITION].to_numpy())))))


# ## Pgen of recurrent TCR

# In[ ]:


recurrent_tcrab = pd.DataFrame(list(map(lambda x: x[0].split("=") + [x[1]], Counter(list(map('='.join, tcr_df[
    tcr_df['recurrent_tcrs_alpha_beta_disease'] != '-'
].loc[:,TRAB_DEFINITION].to_numpy()))).items())))
recurrent_tcrab['disease'] = list(tcr_df[
    tcr_df['recurrent_tcrs_alpha_beta_disease'] != '-'
].groupby(TRAB_DEFINITION).agg({'recurrent_tcrs_alpha_beta_disease': lambda x: list(x)[0]})['recurrent_tcrs_alpha_beta_disease'])

recurrent_tcrab['disease'] = list(tcr_df[
    tcr_df['recurrent_tcrs_alpha_beta_disease'] != '-'
].groupby(TRAB_DEFINITION).agg({'recurrent_tcrs_alpha_beta_disease': lambda x: list(x)[0]})['recurrent_tcrs_alpha_beta_disease'])


recurrent_tcrab['TRA_Pgen'] = list(tcr_df[
    tcr_df['recurrent_tcrs_alpha_beta_disease'] != '-'
].groupby(TRAB_DEFINITION).agg({'TRA_Pgen': lambda x: list(x)[0]})['TRA_Pgen'])

recurrent_tcrab['TRB_Pgen'] = list(tcr_df[
    tcr_df['recurrent_tcrs_alpha_beta_disease'] != '-'
].groupby(TRAB_DEFINITION).agg({'TRB_Pgen': lambda x: list(x)[0]})['TRB_Pgen'])

recurrent_tcrab['individual'] = list(tcr_df[
    tcr_df['recurrent_tcrs_alpha_beta_disease'] != '-'
].groupby(TRAB_DEFINITION).agg({'individual': lambda x: len(np.unique(x))})['individual'])

recurrent_tcrb = pd.DataFrame(list(map(lambda x: x[0].split("=") + [x[1]], Counter(list(map('='.join, tcr_df[
    tcr_df['recurrent_tcrs_beta_disease'] != '-'
].loc[:,TRB_DEFINITION].to_numpy()))).items())))
recurrent_tcrb['disease'] = list(tcr_df[
    tcr_df['recurrent_tcrs_beta_disease'] != '-'
].groupby(TRB_DEFINITION).agg({'recurrent_tcrs_beta_disease': lambda x: list(x)[0]})['recurrent_tcrs_beta_disease'])

recurrent_tcrb['TRA_Pgen'] = list(tcr_df[
    tcr_df['recurrent_tcrs_beta_disease'] != '-'
].groupby(TRB_DEFINITION).agg({'TRA_Pgen': lambda x: list(x)[0]})['TRA_Pgen'])


recurrent_tcrb['TRB_Pgen'] = list(tcr_df[
    tcr_df['recurrent_tcrs_beta_disease'] != '-'
].groupby(TRB_DEFINITION).agg({'TRB_Pgen': lambda x: list(x)[0]})['TRB_Pgen'])

recurrent_tcrb['individual'] = list(tcr_df[
    tcr_df['recurrent_tcrs_beta_disease'] != '-'
].groupby(TRB_DEFINITION).agg({'individual': lambda x: len(np.unique(x))})['individual'])

recurrent_tcra = pd.DataFrame(list(map(lambda x: x[0].split("=") + [x[1]], Counter(list(map('='.join, tcr_df[
    tcr_df['recurrent_tcrs_alpha_disease'] != '-'
].loc[:,TRA_DEFINITION].to_numpy()))).items())))
recurrent_tcra['disease'] = list(tcr_df[
    tcr_df['recurrent_tcrs_alpha_disease'] != '-'
].groupby(TRA_DEFINITION).agg({'recurrent_tcrs_alpha_disease': lambda x: list(x)[0]})['recurrent_tcrs_alpha_disease'])

recurrent_tcra['TRA_Pgen'] = list(tcr_df[
    tcr_df['recurrent_tcrs_alpha_disease'] != '-'
].groupby(TRA_DEFINITION).agg({'TRA_Pgen': lambda x: list(x)[0]})['TRA_Pgen'])


recurrent_tcra['TRB_Pgen'] = list(tcr_df[
    tcr_df['recurrent_tcrs_alpha_disease'] != '-'
].groupby(TRA_DEFINITION).agg({'TRB_Pgen': lambda x: list(x)[0]})['TRB_Pgen'])


recurrent_tcra['individual'] = list(tcr_df[
    tcr_df['recurrent_tcrs_alpha_disease'] != '-'
].groupby(TRA_DEFINITION).agg({'individual': lambda x: len(np.unique(x))})['individual'])


# ### [Figure 3C]

# In[200]:


import warnings
from scipy.stats import sem

warnings.filterwarnings('ignore')
fig,ax=createSubplots(2,1)

tcr_df_groupby_public_alpha = tcr_df
tcr_df_groupby_public_alpha['TRA_Pgen_log'] = -np.log10(tcr_df_groupby_public_alpha['TRA_Pgen'])
tcr_df_groupby_public_alpha['TRB_Pgen_log'] = -np.log10(tcr_df_groupby_public_alpha['TRB_Pgen'])

tcr_df_groupby_public_beta = tcr_df
tcr_df_groupby_public_beta['TRA_Pgen_log'] = -np.log10(tcr_df_groupby_public_beta['TRA_Pgen'])
tcr_df_groupby_public_beta['TRB_Pgen_log'] = -np.log10(tcr_df_groupby_public_beta['TRB_Pgen'])
recurrent_tra_pgen = list(filter(lambda x: not np.isinf(x),
    tcr_df_groupby_public_alpha[
        tcr_df_groupby_public_alpha['recurrent_tcrs_alpha_n_individuals'] > 0
    ]['TRA_Pgen_log']
))
print("Mean of recurrent TCRa generation probability: ", np.mean(
    recurrent_tra_pgen
), sem(recurrent_tra_pgen))
sns.distplot(
    recurrent_tra_pgen,
    ax=ax[0],
    kde_kws=dict(
        bw_adjust=4
    ),
    color = '#1D7293'
)
non_recurrent_tra_pgen = list(filter(lambda x: not np.isinf(x),
    tcr_df_groupby_public_alpha[
        tcr_df_groupby_public_alpha['recurrent_tcrs_alpha_n_individuals'] == 0
    ]['TRA_Pgen_log']
))
print("Mean of non-recurrent TCRa generation probability: ", np.mean(
    non_recurrent_tra_pgen
), sem(non_recurrent_tra_pgen))

sns.distplot(
    non_recurrent_tra_pgen,
    ax=ax[0],
    kde_kws=dict(
        bw_adjust=4
    ),
    color = '#1D2D44'
)

recurrent_trb_pgen = list(filter(lambda x: not np.isinf(x),
    tcr_df_groupby_public_beta[
        tcr_df_groupby_public_beta['recurrent_tcrs_beta_n_individuals'] > 0
    ]['TRB_Pgen_log']
))

print("Mean of recurrent TCRb generation probability: ", np.mean(
    recurrent_trb_pgen
), sem(recurrent_trb_pgen))


sns.distplot(
    recurrent_trb_pgen,
    ax=ax[1],
    kde_kws=dict(
        bw_adjust=4
    ),
    color = '#1D7293'
)

non_recurrent_trb_pgen = list(filter(lambda x: not np.isinf(x),
    tcr_df_groupby_public_beta[
        tcr_df_groupby_public_beta['recurrent_tcrs_beta_n_individuals'] == 0
    ]['TRB_Pgen_log']
))
print("Mean of non-recurrent TCRb generation probability: ", np.mean(
non_recurrent_trb_pgen
), sem(non_recurrent_trb_pgen))



sns.distplot(
    non_recurrent_trb_pgen ,
    ax=ax[1],
    kde_kws=dict(
        bw_adjust=4
    ),
    color = '#1D2D44'
)

ax[0].set_xbound(4,20)
ax[1].set_xbound(5,20)

ax[0].set_title("TRA Pgen Recurrent vs. Non-Recurrent")
ax[1].set_title("TRB Pgen Recurrent vs. Non-Recurrent")
fig.set_size_inches(3,5)
fig.savefig("./figures/recurrent_tcr_pgen.pdf")


# In[196]:


fig,ax=plt.subplots(2,1)
fig.set_size_inches(5,10)

annotation_expansion_info_groupby_public_ab = annotation_expansion_info
annotation_expansion_info_groupby_public_ab['TRA_Pgen_log'] = -np.log10(annotation_expansion_info_groupby_public_ab['TRA_Pgen'])
annotation_expansion_info_groupby_public_ab['TRB_Pgen_log'] = -np.log10(annotation_expansion_info_groupby_public_ab['TRB_Pgen'])

data = list(filter(lambda x: not np.isinf(x[0]) and not np.isinf(x[1]),
        zip(annotation_expansion_info_groupby_public_ab[
            annotation_expansion_info_groupby_public_ab['recurrent_tcrs_alpha_beta_n_individuals'] ==  0
        ]['TRA_Pgen_log'],
            annotation_expansion_info_groupby_public_ab[
            annotation_expansion_info_groupby_public_ab['recurrent_tcrs_alpha_beta_n_individuals'] ==  0
        ]['TRB_Pgen_log'])
))
print(np.array(data).mean(0))


sns.kdeplot(
    x= list(map(lambda x: x[0], data)),
    y= list(map(lambda x: x[1], data)),
    cmap=make_colormap(['#F7F7F7','#1D7293']), shade=False, bw_adjust=1.5,
    ax=ax[0]
)

data = list(filter(lambda x: not np.isinf(x[0]) and not np.isinf(x[1]),
        zip(annotation_expansion_info_groupby_public_ab[
            annotation_expansion_info_groupby_public_ab['recurrent_tcrs_beta_n_individuals'] > 1
        ]['TRA_Pgen_log'],
            annotation_expansion_info_groupby_public_ab[
            annotation_expansion_info_groupby_public_ab['recurrent_tcrs_beta_n_individuals'] > 1
        ]['TRB_Pgen_log'])
))
print(np.array(data).mean(0))

sns.kdeplot(
    x= list(map(lambda x: x[0], data)),
    y= list(map(lambda x: x[1], data)),
    cmap=make_colormap(['#F7F7F7','#1D2D44']), shade=False, bw_adjust=1.5,
    ax=ax[1]
)


ax[0].set_xbound(3,15)
ax[1].set_xbound(3,15)
ax[0].set_ybound(3,20)
ax[1].set_ybound(3,20)

plt.savefig('./figures/20230206_recurrent_tcr_alpha_beta_pgen.pdf')


# ## CDR3mr length of recurrent TCR

# ### [Figure 4D]

# In[202]:


fig,ax=plt.subplots(2,1)
print("Length of CDR3a middle region for recurrent TCRa:", np.mean(list(filter(lambda x: not np.isinf(x),
        tcr_df_groupby_public_alpha[
            tcr_df_groupby_public_alpha['recurrent_tcrs_alpha_n_individuals'] > 0
        ]['CDR3a_mr_length']
    ))),sem(list(filter(lambda x: not np.isinf(x),
        tcr_df_groupby_public_alpha[
            tcr_df_groupby_public_alpha['recurrent_tcrs_alpha_n_individuals'] > 0
        ]['CDR3a_mr_length']
    ))))


sns.distplot(
    list(filter(lambda x: not np.isinf(x),
        tcr_df_groupby_public_alpha[
            tcr_df_groupby_public_alpha['recurrent_tcrs_alpha_n_individuals'] > 0
        ]['CDR3a_mr_length']
    )),
    ax=ax[0],
    kde_kws=dict(
        bw_adjust=5
    ),    
    bins=10, 
    color = '#1D7293'
)

print("Length of CDR3a middle region for recurrent TCRa:", np.mean(list(filter(lambda x: not np.isinf(x),
        tcr_df_groupby_public_alpha[
            tcr_df_groupby_public_alpha['recurrent_tcrs_alpha_n_individuals'] == 0
        ]['CDR3a_mr_length']
    ))),sem(list(filter(lambda x: not np.isinf(x),
        tcr_df_groupby_public_alpha[
            tcr_df_groupby_public_alpha['recurrent_tcrs_alpha_n_individuals'] == 0
        ]['CDR3a_mr_length']
    ))))


sns.distplot(
    list(filter(lambda x:not np.isinf(x),
        tcr_df_groupby_public_alpha[
            tcr_df_groupby_public_alpha['recurrent_tcrs_alpha_n_individuals'] ==  0
        ]['CDR3a_mr_length']
    )),
    ax=ax[0],
    kde_kws=dict(
        bw_adjust=5
    ),    
    bins=20,
    color = '#1D2D44'
)



tcr_df_groupby_public_beta = tcr_df
tcr_df_groupby_public_beta['TRB_Pgen_log'] = -np.log10(tcr_df_groupby_public_beta['TRB_Pgen'])

print("Length of CDR3a middle region for recurrent TCRb:", np.mean(list(filter(lambda x: not np.isinf(x),
        tcr_df_groupby_public_beta[
            tcr_df_groupby_public_beta['recurrent_tcrs_beta_n_individuals'] > 0
        ]['CDR3b_mr_length']
    ))),sem(list(filter(lambda x: not np.isinf(x),
        tcr_df_groupby_public_beta[
            tcr_df_groupby_public_beta['recurrent_tcrs_beta_n_individuals'] > 0
        ]['CDR3b_mr_length']
    ))))

sns.distplot(
    list(filter(lambda x: not np.isinf(x),
        tcr_df_groupby_public_beta[
            tcr_df_groupby_public_beta['recurrent_tcrs_beta_n_individuals'] > 0
        ]['CDR3b_mr_length']
    )),
    ax=ax[1],
    kde_kws=dict(
        bw_adjust=5
    ),
    bins=10,
    color = '#1D7293'
)

print("Length of CDR3a middle region for recurrent TCRb:", np.mean(list(filter(lambda x: not np.isinf(x),
        tcr_df_groupby_public_beta[
            tcr_df_groupby_public_beta['recurrent_tcrs_beta_n_individuals'] == 0
        ]['CDR3b_mr_length']
    ))),sem(list(filter(lambda x: not np.isinf(x),
        tcr_df_groupby_public_beta[
            tcr_df_groupby_public_beta['recurrent_tcrs_beta_n_individuals'] == 0
        ]['CDR3b_mr_length']
    ))))

sns.distplot(
    list(filter(lambda x:not np.isinf(x),
        tcr_df_groupby_public_beta[
            tcr_df_groupby_public_beta['recurrent_tcrs_beta_n_individuals'] ==  0
        ]['CDR3b_mr_length']
    )),
    ax=ax[1],
    kde_kws=dict(
        bw_adjust=5
    ),
    bins=20,
    color = '#1D2D44'
)


ax[0].set_xbound(0,10)
ax[1].set_xbound(0,14)
# ax[0].set_ybound(0,1)
# ax[1].set_ybound(0,1)
fig.set_size_inches(3,5)
fig.savefig("./figures/recurrent_tcr_length.pdf")


# ## Disease specificity analysis of recurrent TCR

# ### [Figure 4E-G]

# In[74]:


m = {'Basal cell carinoma tumor': 'Solid tumor',
 'Large granular lymphocyte leukemia': 'T-LGLL',
 'Nasopharyngeal carcinoma': 'Solid tumor',
 'Healthy': 'Healthy',
 'Clear cell renal cell carcinoma': 'Solid tumor',
 'Esophagus squamous cell carcinoma': 'Solid tumor',
 'CPI-colitis': 'CPI-irAE',
 'COVID-19': 'COVID-19',
 'AML': 'AML',
 'Ulcerative colitis': 'Inflammation',
 'Kawasaki disease': 'Inflammation',
 'Melanoma': 'Solid tumor',
 'Breast cancer': 'Solid tumor',
 'Squamous cell carcinoma tumor': 'Solid tumor',
 'Psoriatic arthritis': 'Inflammation',
 'Arthritis': 'CPI-irAE',
 'Ankylosing spondylitis': 'Inflammation',
 'Metastatic colorectal cancer': 'Solid tumor',
 'Chronic pancreatitis': 'Inflammation',
 'MIS-C': 'COVID-19'}
tcr_df['disease_type_1'] = list(map(m.get, tcr_df['disease']))
m = {'Basal cell carinoma tumor': 'Solid tumor',
 'Large granular lymphocyte leukemia': 'T-LGLL',
 'Nasopharyngeal carcinoma': 'Solid tumor',
 'Healthy': 'Healthy',
 'Clear cell renal cell carcinoma': 'Solid tumor',
 'Esophagus squamous cell carcinoma': 'Solid tumor',
 'CPI-colitis': 'Colitis',
 'COVID-19': 'COVID-19',
 'AML': 'AML',
 'Ulcerative colitis': 'Colitis',
 'Kawasaki disease': 'Kawasaki disease',
 'Melanoma': 'Solid tumor',
 'Breast cancer': 'Solid tumor',
 'Squamous cell carcinoma tumor': 'Solid tumor',
 'Psoriatic arthritis': 'Arthritis',
 'Arthritis': 'Arthritis',
 'Ankylosing spondylitis': 'Ankylosing spondylitis',
 'Metastatic colorectal cancer': 'Solid tumor',
 'Chronic pancreatitis': 'Chronic pancreatitis',
 'MIS-C': 'COVID-19'}
tcr_df['disease_type_2'] = list(map(m.get, tcr_df['disease']))


# In[76]:


disease_label = 'disease_type_1'
from upsetplot import from_memberships
import upsetplot
from itertools import combinations
all_diseases = list(np.unique(tcr_df[disease_label]))
tcr_df['tcr_no_individual'] = list(map(lambda x: '-'.join(x), 
     tcr_df.loc[:,TRAB_DEFINITION].to_numpy()
))
_tmp_df = tcr_df[
    tcr_df['recurrent_tcrs_alpha_beta_n_individuals'] > 1
].groupby(['tcr_no_individual','recurrent_tcrs_alpha_beta_n_individuals']).agg({
    disease_label: list, 'count': len, 'individual': list
})
_tmp_df = _tmp_df[list(map(lambda x: type(x) == list, _tmp_df[disease_label]))]
_agg = pandas_aggregation_to_wide(
    pd.DataFrame([
            list(map(lambda x: x[-1], _tmp_df.index)), 
            list(map(lambda x: ','.join(np.unique(x)), 
            _tmp_df[disease_label])), 
            _tmp_df['count']
        ], index=['recurrent_tcrs_alpha_beta_n_individuals',disease_label,'count']
    ).T.groupby(
    [disease_label]
).agg({
    "count":sum
}))

# _agg = _agg[_agg['recurrent_tcrs_alpha_beta_n_individuals'] > 1]

upset_df = from_memberships(list(map(lambda x: x.split(","), _agg[disease_label])),data=list(_agg['count']))

upset = upsetplot.plot(
    upset_df 
)
upset['intersections'].clear()
_tmp_df = pandas_aggregation_to_wide(_tmp_df)
d = dict(zip(
    _agg[disease_label], 
    list(map(lambda x: list(np.zeros(len(all_diseases), dtype=np.uint32)), _agg[disease_label]))
))
for dt in _tmp_df[_tmp_df['recurrent_tcrs_alpha_beta_n_individuals'] > 1][disease_label]:
    k = ','.join(np.unique(dt))
    for t in dt:
        if k in d.keys():
            d[k][all_diseases.index(t)] += 1
_tmp = []
for i in range(len(upset_df.index.names)):
    _tmp.append((upset_df.index.names[i], upset_df[
        list(filter(lambda x: x[i] == True, upset_df.index.tolist()))
    ].sum()))
names = list(filter(lambda z : z in d.keys(), list(map(lambda x: x[0] ,sorted(_tmp,key=lambda x: -x[1])))))
_names = list(map(lambda x: x[0] ,sorted(_tmp,key=lambda x: -x[1])))
indices = list(names)
indices += list(filter(lambda z: z, map(
    lambda x: list(filter(lambda z: len(set(x).intersection(set(z.split(',')))) == len(x) == len(z.split(',')), d.keys()))[0] 
    if any(map(lambda z: len(set(x).intersection(set(z.split(',')))) == len(x) == len(z.split(',')), d.keys()))
    else None,
    FLATTEN([
        sorted(list(combinations(_names[::-1],q)),
           key=lambda x: [_names.index(z) for z in x]
    ) for q in range(2,7)
    ])
)))
# print(indices)
pd.DataFrame(d, index=all_diseases).T.loc[
   indices
].astype(np.int).plot(
    kind='bar',stacked=True,
    color=list(map(lambda x: disease_type_palette[x],all_diseases)),
    ax=upset['intersections']
)
plt.savefig("./figures/recurrent_tcr_alpha_beta_disease_type.pdf")


# In[78]:


from upsetplot import from_memberships
import upsetplot
from itertools import combinations
disease_label = 'disease_type_1'
all_diseases = list(np.unique(tcr_df[disease_label]))
tcr_df['tcr_no_individual'] = list(map(lambda x: '-'.join(x), 
     tcr_df.loc[:,TRB_DEFINITION].to_numpy()
))
_tmp_df = tcr_df[
    tcr_df['recurrent_tcrs_beta_n_individuals'] > 1
].groupby(['tcr_no_individual','recurrent_tcrs_beta_n_individuals']).agg({
    disease_label: list, 'count': len, 'individual': list
})
_tmp_df = _tmp_df[list(map(lambda x: type(x) == list, _tmp_df[disease_label]))]
_agg = pandas_aggregation_to_wide(
    pd.DataFrame([
            list(map(lambda x: x[-1], _tmp_df.index)), 
            list(map(lambda x: ','.join(np.unique(x)), 
            _tmp_df[disease_label])), 
            _tmp_df['count']
        ], index=['recurrent_tcrs_beta_n_individuals',disease_label,'count']
    ).T.groupby(
    [disease_label]
).agg({
    "count":sum
}))

# _agg = _agg[_agg['recurrent_tcrs_beta_n_individuals'] > 1]

upset_df = from_memberships(list(map(lambda x: x.split(","), _agg[disease_label])),data=list(_agg['count']))

upset = upsetplot.plot(
    upset_df 
)
upset['intersections'].clear()
_tmp_df = pandas_aggregation_to_wide(_tmp_df)
d = dict(zip(
    _agg[disease_label], 
    list(map(lambda x: list(np.zeros(len(all_diseases), dtype=np.uint32)), _agg['disease_type_1']))
))
for dt in _tmp_df[_tmp_df['recurrent_tcrs_beta_n_individuals'] > 1]['disease_type_1']:
    k = ','.join(np.unique(dt))
    for t in dt:
        if k in d.keys():
            d[k][all_diseases.index(t)] += 1
_tmp = []
for i in range(len(upset_df.index.names)):
    _tmp.append((upset_df.index.names[i], upset_df[
        list(filter(lambda x: x[i] == True, upset_df.index.tolist()))
    ].sum()))
names = list(filter(lambda z : z in d.keys(), list(map(lambda x: x[0] ,sorted(_tmp,key=lambda x: -x[1])))))
_names = list(map(lambda x: x[0] ,sorted(_tmp,key=lambda x: -x[1])))
indices = list(names)
indices += list(filter(lambda z: z, map(
    lambda x: list(filter(lambda z: len(set(x).intersection(set(z.split(',')))) == len(x) == len(z.split(',')), d.keys()))[0] 
    if any(map(lambda z: len(set(x).intersection(set(z.split(',')))) == len(x) == len(z.split(',')), d.keys()))
    else None,
    FLATTEN([
        sorted(list(combinations(_names[::-1],q)),
           key=lambda x: [_names.index(z) for z in x]
    ) for q in range(2,7)
    ])
)))
# print(indices)
pd.DataFrame(d, index=all_diseases).T.loc[
   indices
].astype(np.int).plot(
    kind='bar',stacked=True,
    color=list(map(lambda x: disease_type_palette[x],all_diseases)),
    ax=upset['intersections']
)
plt.savefig("./figures/recurrent_tcr_beta_disease.pdf")


# In[80]:


from upsetplot import from_memberships
import upsetplot
from itertools import combinations
all_diseases = list(np.unique(tcr_df['disease_type_1']))
tcr_df['tcr_no_individual'] = list(map(lambda x: '-'.join(x), 
     tcr_df.loc[:,TRA_DEFINITION].to_numpy()
))
_tmp_df = tcr_df[
    tcr_df['recurrent_tcrs_alpha_n_individuals'] > 1
].groupby(['tcr_no_individual','recurrent_tcrs_alpha_n_individuals']).agg({
    "disease_type_1": list, 'count': len, 'individual': list
})
_agg = pandas_aggregation_to_wide(
    pd.DataFrame([
            list(map(lambda x: x[-1], _tmp_df.index)), 
            list(map(lambda x: ','.join(np.unique(x)), 
            _tmp_df['disease_type_1'])), 
            _tmp_df['count']
        ], index=['recurrent_tcrs_alpha_n_individuals','disease_type_1','count']
    ).T.groupby(
    ['disease_type_1']
).agg({
    "count":sum
}))

# _agg = _agg[_agg['recurrent_tcrs_alpha_n_individuals'] > 1]

upset_df = from_memberships(list(map(lambda x: x.split(","), _agg['disease_type_1'])),data=list(_agg['count']))

upset = upsetplot.plot(
    upset_df 
)
upset['intersections'].clear()
_tmp_df = pandas_aggregation_to_wide(_tmp_df)
d = dict(zip(
    _agg['disease_type_1'], 
    list(map(lambda x: list(np.zeros(len(all_diseases), dtype=np.uint32)), _agg['disease_type_1']))
))
for dt in _tmp_df[_tmp_df['recurrent_tcrs_alpha_n_individuals'] > 1]['disease_type_1']:
    k = ','.join(np.unique(dt))
    for t in dt:
        if k in d.keys():
            d[k][all_diseases.index(t)] += 1
_tmp = []
for i in range(len(upset_df.index.names)):
    _tmp.append((upset_df.index.names[i], upset_df[
        list(filter(lambda x: x[i] == True, upset_df.index.tolist()))
    ].sum()))
names = list(filter(lambda z : z in d.keys(), list(map(lambda x: x[0] ,sorted(_tmp,key=lambda x: -x[1])))))
_names = list(map(lambda x: x[0] ,sorted(_tmp,key=lambda x: -x[1])))
indices = list(names)
indices += list(filter(lambda z: z, map(
    lambda x: list(filter(lambda z: len(set(x).intersection(set(z.split(',')))) == len(x) == len(z.split(',')), d.keys()))[0] 
    if any(map(lambda z: len(set(x).intersection(set(z.split(',')))) == len(x) == len(z.split(',')), d.keys()))
    else None,
    FLATTEN([
        sorted(list(combinations(_names[::-1],q)),
           key=lambda x: [_names.index(z) for z in x]
    ) for q in range(2,7)
    ])
)))
# print(indices)
pd.DataFrame(d, index=all_diseases).T.loc[
   indices
].astype(np.int).plot(
    kind='bar',stacked=True,
    color=list(map(lambda x: disease_type_palette[x],all_diseases)),
    ax=upset['intersections']
)
plt.savefig("./figures/recurrent_tcr_alpha_disease.pdf")


# # TCR Coherence analysis

# In[1]:


import pickle
mismatch_pairing = {}
for i in ['CD8','CD4','Treg','Naive_CD8','Naive_CD4']:
    with open(f"./data/20230201_mismatch_pairing_{i}.pkl", "rb") as f:
        mismatch_pairing[i] = pickle.load(f)


# In[11]:


from scipy.stats import entropy, sem
from functools import cache
palette = reannotated_prediction_palette
for n_pairing in range(3,6):
    @cache
    def func(k,m):
        X = mismatch_pairing[k]['beta_alpha_mismatched_pairing'][m]
        X = X[np.array((X.sum(1) > 1)).flatten()]
        tmp_df = pd.DataFrame(list(map(str, X)))
        tmp_df['index'] = range(len(tmp_df))
        tmp_agg = tmp_df.groupby(0).agg({"index":lambda x: list(x)[0]})
        tmp_agg['n_pairing'] = list(map(lambda x: len(x.split("\t"))-1, tmp_agg.index))
        return tmp_agg[tmp_agg['n_pairing'] == n_pairing]['index'].to_numpy().flatten()

    entropy_mean_arr = pd.DataFrame([[np.mean(
        np.array(mismatch_pairing[k]['statistics']['beta_alpha_coherence'][m]['entropys'])[func(k,m)].flatten()
    ) for k in mismatch_pairing.keys()] for m in range(6)], columns=mismatch_pairing.keys())
    entropy_err_arr = pd.DataFrame([[sem(
        np.array(mismatch_pairing[k]['statistics']['beta_alpha_coherence'][m]['entropys'])[func(k,m)].flatten()
    ) for k in mismatch_pairing.keys()] for m in range(6)], columns=mismatch_pairing.keys())


    percentage_mean_arr = pd.DataFrame([[np.mean(
        np.array(mismatch_pairing[k]['statistics']['beta_alpha_coherence'][m]['max_count_percentages'])[func(k,m)].flatten()
    ) for k in mismatch_pairing.keys()] for m in range(6)], columns=mismatch_pairing.keys())
    percentage_err_arr = pd.DataFrame([[sem(
        np.array(mismatch_pairing[k]['statistics']['beta_alpha_coherence'][m]['max_count_percentages'])[func(k,m)].flatten()
    ) for k in mismatch_pairing.keys()] for m in range(6)], columns=mismatch_pairing.keys())

    percentage_mean_arr = percentage_mean_arr.fillna(0)
    percentage_err_arr = percentage_err_arr.fillna(0)
    entropy_mean_arr = entropy_mean_arr.fillna(0)
    entropy_err_arr = entropy_err_arr.fillna(0)
    
    fig,axes=createSubplots(1,2)
    fig.set_size_inches(10,5)
    for i in palette.keys(): 
        j = i.replace(" ", "_")
        if j in percentage_mean_arr.columns:
            axes[0].errorbar(
                x=range(6),
                y=percentage_mean_arr.loc[:,j].to_numpy().flatten(), 
                yerr=percentage_err_arr.loc[:,j].to_numpy().flatten(), 
                color=palette[i]
            )
            axes[0].scatter(
                x=range(6),
                y=percentage_mean_arr.loc[:,j].to_numpy().flatten(),
                c=palette[i]
            )
    # fig.savefig("/Users/snow/Desktop/20220214_beta_alpha_coherence.pdf")
    # mean_arr.to_csv("/Users/snow/Desktop/2023-TCR-huARdb-Analysis-Paper-Figures/Data/Notebook/data/20220214_beta_alpha_coherence_entropy_mean.csv")


    for i in palette.keys(): 
        j = i.replace(" ", "_")
        if j in entropy_mean_arr.columns:
            axes[1].errorbar(
                x=range(6),
                y=entropy_mean_arr.loc[:,j].to_numpy().flatten(), 
                yerr=entropy_err_arr.loc[:,j].to_numpy().flatten(), 
                color=palette[i]
            )
            axes[1].scatter(
                x=range(6),
                y=entropy_mean_arr.loc[:,j].to_numpy().flatten(),
                c=palette[i]
            )
    fig.savefig(f"./data/20220214_mismatched_pairing/20220214_beta_alpha_coherence_n_pairing_{n_pairing}.pdf")
    percentage_mean_arr.to_csv(f"./data/20220214_mismatched_pairing/20220214_beta_alpha_coherence_percentage_mean_n_pairing_{n_pairing}.csv")
    percentage_err_arr.to_csv(f"./data/20220214_mismatched_pairing/20220214_beta_alpha_coherence_percentage_err_n_pairing_{n_pairing}.csv")
    plt.close()


# In[14]:


from scipy.stats import entropy, sem
from functools import cache
n_pairing='average'
@cache
def func(k,m):
    X = mismatch_pairing[k]['alpha_beta_mismatched_pairing'][m]
    X = X[np.array((X.sum(1) > 1)).flatten()]
    tmp_df = pd.DataFrame(list(map(str, X)))
    tmp_df['index'] = range(len(tmp_df))
    tmp_agg = tmp_df.groupby(0).agg({"index":lambda x: list(x)[0]})
    tmp_agg['n_pairing'] = list(map(lambda x: len(x.split("\t"))-1, tmp_agg.index))
    return tmp_agg['index'].to_numpy().flatten()

entropy_mean_arr = pd.DataFrame([[np.mean(
    np.array(mismatch_pairing[k]['statistics']['alpha_beta_coherence'][m]['entropys'])[func(k,m)].flatten()
) for k in mismatch_pairing.keys()] for m in range(6)], columns=mismatch_pairing.keys())
entropy_err_arr = pd.DataFrame([[sem(
    np.array(mismatch_pairing[k]['statistics']['alpha_beta_coherence'][m]['entropys'])[func(k,m)].flatten()
) for k in mismatch_pairing.keys()] for m in range(6)], columns=mismatch_pairing.keys())


percentage_mean_arr = pd.DataFrame([[np.mean(
    np.array(mismatch_pairing[k]['statistics']['alpha_beta_coherence'][m]['max_count_percentages'])[func(k,m)].flatten()
) for k in mismatch_pairing.keys()] for m in range(6)], columns=mismatch_pairing.keys())
percentage_err_arr = pd.DataFrame([[sem(
    np.array(mismatch_pairing[k]['statistics']['alpha_beta_coherence'][m]['max_count_percentages'])[func(k,m)].flatten()
) for k in mismatch_pairing.keys()] for m in range(6)], columns=mismatch_pairing.keys())

percentage_mean_arr = percentage_mean_arr.fillna(0)
percentage_err_arr = percentage_err_arr.fillna(0)
entropy_mean_arr = entropy_mean_arr.fillna(0)
entropy_err_arr = entropy_err_arr.fillna(0)

fig,axes=createSubplots(1,2)
fig.set_size_inches(10,5)
for i in palette.keys(): 
    j = i.replace(" ", "_")
    if j in percentage_mean_arr.columns:
        axes[0].errorbar(
            x=range(6),
            y=percentage_mean_arr.loc[:,j].to_numpy().flatten(), 
            yerr=percentage_err_arr.loc[:,j].to_numpy().flatten(), 
            color=palette[i]
        )
        axes[0].scatter(
            x=range(6),
            y=percentage_mean_arr.loc[:,j].to_numpy().flatten(),
            c=palette[i]
        )
        
        
for i in palette.keys(): 
    j = i.replace(" ", "_")
    if j in percentage_mean_arr.columns:
        axes[1].errorbar(
            x=range(6),
            y=entropy_mean_arr.loc[:,j].to_numpy().flatten(), 
            yerr=entropy_err_arr.loc[:,j].to_numpy().flatten(), 
            color=palette[i]
        )
        axes[1].scatter(
            x=range(6),
            y=entropy_mean_arr.loc[:,j].to_numpy().flatten(),
            c=palette[i]
        )


fig.savefig(f"./data//20220214_mismatched_pairing/20220214_alpha_beta_coherence_n_pairing_{n_pairing}.pdf")
percentage_mean_arr.to_csv(f"./data/20220214_mismatched_pairing/20220214_alpha_beta_coherence_percentage_mean_n_pairing_{n_pairing}.csv")
percentage_err_arr.to_csv(f"./data/20220214_mismatched_pairing/20220214_alpha_beta_coherence_percentage_err_n_pairing_{n_pairing}.csv")
plt.close()


# # Matching to known TCRpMHC dataset

# In[1]:


known_tcrpmhc = pd.read_csv("./data/sidhom_tcrbert_10x_covid10x_merged_dataset.csv")
known_tcrpmhc
peptide2epitope = {}
unknownepitopename = set()
for k,v in zip(known_tcrpmhc['peptide'], known_tcrpmhc['epitope']):
    if v != '-':
        peptide2epitope[k] = v
for k,v in zip(known_tcrpmhc['peptide'], known_tcrpmhc['epitope']):
    if v == '-':
        unknownepitopename.add(k)


# In[2]:


known_tcrpmhc


# In[32]:


len(np.unique(known_tcrpmhc['peptide']))


# In[9]:


peptide_mapping = {}
for a,b,p in known_tcrpmhc.loc[:,['CDR3a','CDR3b','peptide']].to_numpy():
    cdr3 = '{}-{}'.format(a,b)
    if cdr3 not in peptide_mapping.keys():
        peptide_mapping[cdr3] = [p]
    else:
        if p not in peptide_mapping[cdr3]:
            peptide_mapping[cdr3].append(p)
print(len(peptide_mapping))
_tmp = []

for i,(a,b) in enumerate(tcr_df.loc[:,['CDR3a','CDR3b']].to_numpy()):
    cdr3 = '{}-{}'.format(a,b)
    if cdr3 in peptide_mapping.keys():
        _tmp.append(','.join(peptide_mapping[cdr3]))
    else:
        _tmp.append('-')
        
tcr_df['alpha_beta_pmhc'] = _tmp

peptide_mapping = {}
for a,b,p in known_tcrpmhc.loc[:,['CDR3a','CDR3b','peptide']].to_numpy():
    cdr3 = a
    if cdr3 not in peptide_mapping.keys():
        peptide_mapping[cdr3] = [p]
    else:
        if p not in peptide_mapping[cdr3]:
            peptide_mapping[cdr3].append(p)
            
            
print(len(peptide_mapping))
_tmp = []
for i,(a,b) in enumerate(tcr_df.loc[:,['CDR3a','CDR3b']].to_numpy()):
    cdr3 = a
    if cdr3 in peptide_mapping.keys():
        _tmp.append(','.join(peptide_mapping[cdr3]))
    else:
        _tmp.append('-')
tcr_df['alpha_pmhc'] = _tmp

peptide_mapping = {}
for a,b,p in known_tcrpmhc.loc[:,['CDR3a','CDR3b','peptide']].to_numpy():
    cdr3 = b
    if cdr3 not in peptide_mapping.keys():
        peptide_mapping[cdr3] = [p]
    else:
        if p not in peptide_mapping[cdr3]:
            peptide_mapping[cdr3].append(p)

print(len(peptide_mapping))
_tmp = []
for i,(a,b) in enumerate(tcr_df.loc[:,['CDR3a','CDR3b']].to_numpy()):
    cdr3 = b
    if cdr3 in peptide_mapping.keys():
        _tmp.append(','.join(peptide_mapping[cdr3]))
    else:
        _tmp.append('-')
tcr_df['beta_pmhc'] = _tmp


# ### [Figure S4B]

# In[15]:


disease_types = list(np.unique(tcr_df['disease_type_1']))
sumstat = {}
for i in np.unique(tcr_df.loc[tcr_df['alpha_beta_pmhc'] != '-', 'alpha_beta_pmhc']):
    c = Counter(tcr_df.loc[tcr_df['alpha_beta_pmhc'] == i, 'disease_type_1'])
    for j in i.split(','):
        if j in sumstat.keys():
            sumstat[j] += np.array([c.get(x,0) for x in disease_types])
        else:
            sumstat[j] = np.array([c.get(x,0) for x in disease_types])
sumstat = pd.DataFrame(sumstat, index=disease_types).T
fig,ax=plt.subplots(1,3)
fig.set_size_inches(10,3)
sumstat.loc[sumstat.sum(1).sort_values(ascending=False).index].iloc[:3,:].plot(kind='bar', stacked=True, color=['#E64B35','#4DBBD5','#029F87','#3C5488','#F39B7F','#8491B4','#91D1C2','#7E6048','#B09C85'],ax=ax[0])
sumstat.loc[sumstat.sum(1).sort_values(ascending=False).index].iloc[3:20,:].plot(kind='bar', stacked=True, color=['#E64B35','#4DBBD5','#029F87','#3C5488','#F39B7F','#8491B4','#91D1C2','#7E6048','#B09C85'],ax=ax[1])
sumstat.loc[sumstat.sum(1).sort_values(ascending=False).index].iloc[20:,:].plot(kind='bar', stacked=True, color=['#E64B35','#4DBBD5','#029F87','#3C5488','#F39B7F','#8491B4','#91D1C2','#7E6048','#B09C85'],ax=ax[2])

fig.savefig("./figures/alpha_beta_mhc_disease_composition.pdf")


# In[16]:


disease_types = list(np.unique(tcr_df['disease_type_1']))
sumstat = {}
for i in np.unique(tcr_df.loc[tcr_df['beta_pmhc'] != '-', 'beta_pmhc']):
    c = Counter(tcr_df.loc[tcr_df['beta_pmhc'] == i, 'disease_type_1'])
    for j in i.split(','):
        if j in sumstat.keys():
            sumstat[j] += np.array([c.get(x,0) for x in disease_types])
        else:
            sumstat[j] = np.array([c.get(x,0) for x in disease_types])
sumstat = pd.DataFrame(sumstat, index=disease_types).T

fig,ax=plt.subplots(1,3)
fig.set_size_inches(20,3)
sumstat.loc[sumstat.sum(1).sort_values(ascending=False).index].iloc[:3,:].plot(kind='bar', stacked=True, color=['#E64B35','#4DBBD5','#029F87','#3C5488','#F39B7F','#8491B4','#91D1C2','#7E6048','#B09C85'],ax=ax[0])
sumstat.loc[sumstat.sum(1).sort_values(ascending=False).index].iloc[3:11,:].plot(kind='bar', stacked=True, color=['#E64B35','#4DBBD5','#029F87','#3C5488','#F39B7F','#8491B4','#91D1C2','#7E6048','#B09C85'],ax=ax[1])
sumstat.loc[sumstat.sum(1).sort_values(ascending=False).index].iloc[11:50,:].plot(kind='bar', stacked=True, color=['#E64B35','#4DBBD5','#029F87','#3C5488','#F39B7F','#8491B4','#91D1C2','#7E6048','#B09C85'],ax=ax[2])
fig.savefig("./figures/beta_mhc_disease_composition.pdf")


# In[17]:


disease_types = list(np.unique(tcr_df['disease_type_1']))
sumstat = {}
for i in np.unique(tcr_df.loc[tcr_df['alpha_pmhc'] != '-', 'alpha_pmhc']):
    c = Counter(tcr_df.loc[tcr_df['alpha_pmhc'] == i, 'disease_type_1'])
    for j in i.split(','):
        if j in sumstat.keys():
            sumstat[j] += np.array([c.get(x,0) for x in disease_types])
        else:
            sumstat[j] = np.array([c.get(x,0) for x in disease_types])
sumstat = pd.DataFrame(sumstat, index=disease_types).T

fig,ax=plt.subplots(1,4)
fig.set_size_inches(24,3)
sumstat.loc[sumstat.sum(1).sort_values(ascending=False).index].iloc[:1,:].plot(kind='bar', stacked=True, color=['#E64B35','#4DBBD5','#029F87','#3C5488','#F39B7F','#8491B4','#91D1C2','#7E6048','#B09C85'],ax=ax[0])
sumstat.loc[sumstat.sum(1).sort_values(ascending=False).index].iloc[1:6,:].plot(kind='bar', stacked=True, color=['#E64B35','#4DBBD5','#029F87','#3C5488','#F39B7F','#8491B4','#91D1C2','#7E6048','#B09C85'],ax=ax[1])
sumstat.loc[sumstat.sum(1).sort_values(ascending=False).index].iloc[6:10].plot(kind='bar', stacked=True, color=['#E64B35','#4DBBD5','#029F87','#3C5488','#F39B7F','#8491B4','#91D1C2','#7E6048','#B09C85'],ax=ax[2])
sumstat.loc[sumstat.sum(1).sort_values(ascending=False).index].iloc[10:30].plot(kind='bar', stacked=True, color=['#E64B35','#4DBBD5','#029F87','#3C5488','#F39B7F','#8491B4','#91D1C2','#7E6048','#B09C85'],ax=ax[3])
fig.savefig("./figures/alpha_mhc_disease_composition.pdf")


# # TCR-DeepInsight Result Visualization

# In[20]:


result_tcr = pd.read_csv("./data//result_disease_specific_tcr.csv")


# In[21]:


result_tcr


# In[24]:


df_covid19 = result_tcr[
    np.array(result_tcr["disease_type"] == 'COVID-19') &
    np.array(result_tcr['tcr_count'] > 3) &
    np.array(result_tcr['number_of_individuals'] > 2) &
    np.array(result_tcr['tcr_similarity_score'] < 600) &
    np.array(result_tcr['disease_specificity_score'] > 0) &
    np.array(result_tcr['cell_type'] != 'MAIT')
].sort_values("number_of_individuals")
plt.ion()
fig,ax=createFig()
ax.scatter(
    df_covid19['tcr_similarity_score'], 
    df_covid19['disease_specificity_score'],
    c=list(map(lambda x: reannotated_prediction_palette.get(x[0]) if x[1]**2 +x[2]**2>300**2 else '#D7D7D7', zip(df_covid19['cell_type'], df_covid19['disease_specificity_score'], df_covid19['tcr_similarity_score']))), 
    s=df_covid19['tcr_count'] * 6, 
    linewidths=0
)


# In[25]:


df_solid_tumor = result_tcr[
    np.array(result_tcr["disease_type"] == 'Solid tumor') &
    np.array(result_tcr['tcr_count'] > 3) &
    np.array(result_tcr['number_of_individuals'] > 2) &
    np.array(result_tcr['tcr_similarity_score'] < 600) &
    np.array(result_tcr['disease_specificity_score'] > 0) &
    np.array(result_tcr['cell_type'] != 'MAIT')
].sort_values("number_of_individuals")
plt.ion()
fig,ax=createFig()
ax.scatter(
    df_solid_tumor['tcr_similarity_score'], 
    df_solid_tumor['disease_specificity_score'],
    c=list(map(lambda x: reannotated_prediction_palette.get(x[0]) if x[1]**2 +x[2]**2>300**2 else '#D7D7D7', zip(df_solid_tumor['cell_type'], df_solid_tumor['disease_specificity_score'], df_solid_tumor['tcr_similarity_score']))), 
    s=df_solid_tumor['tcr_count'] * 6, 
    linewidths=0
)


# In[26]:


df_kd = result_tcr[
    np.array(result_tcr["disease_type"] == 'Kawasaki disease') &
    np.array(result_tcr['tcr_count'] > 3) &
    np.array(result_tcr['number_of_individuals'] > 2) &
    np.array(result_tcr['tcr_similarity_score'] < 600) &
    np.array(result_tcr['disease_specificity_score'] > 0) &
    np.array(result_tcr['cell_type'] != 'MAIT')
].sort_values("number_of_individuals")
plt.ion()
fig,ax=createFig()
ax.scatter(
    df_kd['tcr_similarity_score'], 
    df_kd['disease_specificity_score'],
    c=list(map(lambda x: reannotated_prediction_palette.get(x[0]) if x[1]**2 +x[2]**2>300**2 else '#D7D7D7', zip(df_kd['cell_type'], df_kd['disease_specificity_score'], df_kd['tcr_similarity_score']))), 
    s=df_kd['tcr_count'] * 6, 
    linewidths=0
)


# In[29]:


df_as = result_tcr[
    np.array(result_tcr["disease_type"] == 'Ankylosing spondylitis') &
    np.array(result_tcr['tcr_count'] > 2) &
    np.array(result_tcr['number_of_individuals'] > 1) &
    np.array(result_tcr['tcr_similarity_score'] < 600) &
    np.array(result_tcr['disease_specificity_score'] > 0) &
    np.array(result_tcr['cell_type'] != 'MAIT')
].sort_values("number_of_individuals")
plt.ion()
fig,ax=createFig()
ax.scatter(
    df_as['tcr_similarity_score'], 
    df_as['disease_specificity_score'],
    c=list(map(lambda x: reannotated_prediction_palette.get(x[0]) if x[1]**2 +x[2]**2>300**2 else '#D7D7D7', zip(df_as['cell_type'], df_as['disease_specificity_score'], df_as['tcr_similarity_score']))), 
    s=df_as['tcr_count'] * 6, 
    linewidths=0
)


# In[30]:


df_colitis = result_tcr[
    np.array(result_tcr["disease_type"] == 'Colitis') &
    np.array(result_tcr['tcr_count'] > 2) &
    np.array(result_tcr['number_of_individuals'] > 1) &
    np.array(result_tcr['tcr_similarity_score'] < 600) &
    np.array(result_tcr['disease_specificity_score'] > 0) &
    np.array(result_tcr['cell_type'] != 'MAIT')
].sort_values("number_of_individuals")
plt.ion()
fig,ax=createFig()
ax.scatter(
    df_colitis['tcr_similarity_score'], 
    df_colitis['disease_specificity_score'],
    c=list(map(lambda x: reannotated_prediction_palette.get(x[0]) if x[1]**2 +x[2]**2>300**2 else '#D7D7D7', zip(df_colitis['cell_type'], df_colitis['disease_specificity_score'], df_colitis['tcr_similarity_score']))), 
    s=df_colitis['tcr_count'] * 6, 
    linewidths=0
)

