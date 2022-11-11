import os
import pandas as pd
import numpy as np

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from itertools import cycle

pylab.rcParams['figure.figsize'] = 20, 12

database_info = {'NumOfSubjects': 120, 'NumOfClasses': 10, 'NumOfSamplePerClass': 5}
NumOfSubjects = database_info['NumOfSubjects']
NumOfClasses = database_info['NumOfClasses']
NumOfSamplePerClass = database_info['NumOfSamplePerClass']
NumOfGen = (NumOfSubjects * NumOfSamplePerClass * (NumOfSamplePerClass - 1) / 2)
NumOfImp = (NumOfSamplePerClass ** 2 * NumOfSubjects * (NumOfSubjects - 1) / 2)
print(NumOfGen, NumOfImp)


knuckle_img_dir = '/media/administrator/Data/PythonDir/maskrcnn-benchmark/datasets/knuckle/knuckle_seg_96/'
fp_img_dir = '/media/administrator/Data/data/FingerprintDatabase/'

knuckle_des_score_dir = './knuckle_des_score/'
fp_des_score_dir = './fp_des_score/'
dynamic_score_dir = './dynamic_fusion_score/'
static_score_dir = './static_fusion_score/'
min_score_dir = './min_fusion_score/'
tanh_score_dir = './tanh_fusion_score/'
product_score_dir = './product_fusion_score/'


img_info = {'knuckle': knuckle_img_dir, 'fp': fp_img_dir}
score_info = {'knuckle_des_score_dir': knuckle_des_score_dir,
              'fp_des_score_dir': fp_des_score_dir}
fusion_info = {'knuckle': knuckle_des_score_dir,
               'fingerprint': fp_des_score_dir,
               'dynamic_fusion': dynamic_score_dir,
               'static_fusion': static_score_dir,
               'min_fusion': min_score_dir,
               'tanh_fusion': tanh_score_dir,
               'product_fusion': product_score_dir}


# compute fusion of dynamic, static
class ScoreFusion:
    def __init__(self, img_info, score_info):
        for key, value in img_info.items():
            setattr(self, key, value)
        for key, value in score_info.items():
            setattr(self, key, value)
        self.biometrics = [key for key in img_info.keys()]
        self.files = self.get_files()
        self.fp_namechange()
        self.file_class = self.build_file_class()
        self.n_class = self.get_numofclasses()
        self.classes = self.get_classes()

    def get_file_bybiometric(self, biometric):
        files = []
        img_dir = getattr(self, biometric)
        for (dirpath, dirnames, filenames) in os.walk(img_dir):
            for filename in filenames:
                files.append(filename[:-4])
        files.sort()
        return files

    def get_files(self):
        files = defaultdict(list)
        for biometric in self.biometrics:
            files[biometric] = self.get_file_bybiometric(biometric)
        return files

    def build_file_class(self):
        files = defaultdict(list)
        for biometric in self.biometrics:
            file_class = defaultdict(list)
            for filename in self.files[biometric]:
                _, clas, _ = filename.split('-')
                file_class[clas].append(filename)
            files[biometric] = file_class
        return files

    def fp_namechange(self):
        new_name = []
        for filename in self.files['fp']:
            subject_id, _, clas, instance = filename.split('-')
            name = '-'.join((subject_id, clas, instance))
            new_name.append(name)
        self.files['fp'] = new_name

    def get_numofclasses(self):
        n_class = dict()
        for biometric in self.biometrics:
            n_class[biometric] = len(self.file_class[biometric])

        value_ls = []
        for value in n_class.values():
            value_ls.append(value)
        if len(set(value_ls)) == 1:
            return value_ls[0]

    def get_classes(self):
        classes = []
        for key in self.file_class['fp'].keys():
            classes.append(key)
        return classes

    def save_fusion_score(self, fusion_rule):
        for clas in self.classes:
            #             clas = '01'
            #             if clas == '05' or clas == '06':
            set_k = set(self.file_class['knuckle'][clas])
            set_f = set(self.file_class['fp'][clas])

            common = set_k & set_f
            dif1 = set_k - set_f
            dif2 = set_f - set_k
            union = set_k | set_f

            kf_list = list(union)
            kf_list.sort()

            fs_df = pd.DataFrame(index=kf_list, columns=kf_list)
            fs_df.fillna(-1, inplace=True)

            fp_df_name = ''.join(('_'.join(('fp', 'score', clas)), '.csv'))
            fp_df = pd.read_csv(os.path.join(self.fp_des_score_dir, fp_df_name), header=0, index_col=0)

            knuckle_df_name = ''.join(('_'.join(('knuckle', 'score', clas)), '.csv'))
            knuckle_df = pd.read_csv(os.path.join(self.knuckle_des_score_dir, knuckle_df_name), header=0, index_col=0)

            w_fp, w_kn = 0.5, 0.5
            for i, row in enumerate(tqdm(kf_list)):
                if (row in fp_df.index) and (row in knuckle_df.index):
                    for j, col in enumerate(kf_list[i + 1:]):
                        if (col in fp_df.columns) and (col in knuckle_df.columns):
                            if fusion_rule == 'static':
                                fs_df.loc[row, col] = w_fp * fp_df.loc[row, col] + w_kn * knuckle_df.loc[row, col]
                            elif fusion_rule == 'min':
                                fs_df.loc[row, col] = min(fp_df.loc[row, col], knuckle_df.loc[row, col])
                            elif fusion_rule == 'product':
                                fs_df.loc[row, col] = fp_df.loc[row, col] * knuckle_df.loc[row, col]
                        #                             fs_df.loc[row, col] = w_fp * fp_df.loc[row, col] + w_kn * knuckle_df.loc[row, col]
                        elif (col not in fp_df.columns) and (col in knuckle_df.columns):
                            fs_df.loc[row, col] = knuckle_df.loc[row, col]
                        elif (col in fp_df.columns) and (col not in knuckle_df.columns):
                            fs_df.loc[row, col] = fp_df.loc[row, col]
                elif row not in fp_df.index:
                    for j, col in enumerate(kf_list[i + 1:]):
                        if col in knuckle_df.columns:
                            fp_df.loc[row, col] = 1
                            if fusion_rule == 'static':
                                fs_df.loc[row, col] = w_fp * fp_df.loc[row, col] + w_kn * knuckle_df.loc[row, col]
                            elif fusion_rule == 'min':
                                fs_df.loc[row, col] = min(fp_df.loc[row, col], knuckle_df.loc[row, col])
                            elif fusion_rule == 'product':
                                fs_df.loc[row, col] = fp_df.loc[row, col] * knuckle_df.loc[row, col]
                else:
                    for j, col in enumerate(kf_list[i + 1:]):
                        if col in fp_df.columns:
                            knuckle_df.loc[row, col] = 1
                            if fusion_rule == 'static':
                                fs_df.loc[row, col] = w_fp * fp_df.loc[row, col] + w_kn * knuckle_df.loc[row, col]
                            elif fusion_rule == 'min':
                                fs_df.loc[row, col] = min(fp_df.loc[row, col], knuckle_df.loc[row, col])
                            elif fusion_rule == 'product':
                                fs_df.loc[row, col] = fp_df.loc[row, col] * knuckle_df.loc[row, col]

            des_name = ''.join(('_'.join(('fusion', 'score', clas)), '.csv'))
            score_dir = '_'.join((fusion_rule, 'fusion', 'score'))
            fs_df.to_csv(os.path.join(score_dir, des_name))

            fs_2d = fs_df.to_numpy()
            upper_mask = np.triu_indices(fs_2d.shape[0], 1)
            lower_mask = np.tril_indices(fs_2d.shape[0], 0)
            fs_upper = fs_2d[upper_mask]
            fs_lower = fs_2d[lower_mask]
            assert (fs_lower == -1).all()

            assert (fs_2d.shape[0] ** 2 - fs_2d.shape[0]) / 2 == fs_upper.shape[0]


#             break

sf = ScoreFusion(img_info, score_info)

# run this part only once
fusion_rules = ['static', 'min', 'product']
for fusion_rule in fusion_rules:
    print('Fusion Rule:', fusion_rule)
    sf.save_fusion_score(fusion_rule)


class CSV2ROC:
    def __init__(self, src_csv_dir):
        self.src_csv_dir = src_csv_dir
        self.filenamelist = self.get_all_files()
        self.classes = self.get_classes()
        self.n_classes = len(self.classes)
        self.y, self.scores, self.n_g, self.n_i = self.get_score()
        self.fpr, self.tpr, self.eer, self.roc_auc = self.compute_roc()

    def get_all_files(self):
        filenamelist = os.listdir(self.src_csv_dir)
        filenamelist.sort()
        return filenamelist

    def get_classes(self):
        classes = []
        for filename in self.filenamelist:
            _, _, finger_type = filename[:-4].split('_')
            classes.append(finger_type)
        return classes

    def get_score(self):
        y = dict()
        scores = dict()
        num_g = dict()
        num_i = dict()
        for index, finger_type in zip(range(self.n_classes), self.classes):
            #             index = 0
            #             finger_type = '01'
            csv_path = os.path.join(self.src_csv_dir, self.filenamelist[index])
            df = pd.read_csv(csv_path, header=0)

            # g_s: genuine score
            # i_s: impostor score
            g_s = []
            i_s = []
            n_g, n_i = 0, 0

            for row in df.index:
                can_subject_id, can_finger_id, can_instance = df.iloc[row, 0].split('-')
                for col in range(row + 1, len(df.columns)):
                    if df.iloc[row, col] >= 0:
                        ref_subject_id, ref_finger_id, ref_instance = df.columns[col].split('-')
                        if (can_subject_id == ref_subject_id) and (can_finger_id == ref_finger_id):
                            if (can_instance != ref_instance):
                                #                 print("genuine")
                                n_g += 1
                                g_s.append(df.iloc[row, col])
                        else:
                            #             print("impostor")
                            n_i += 1
                            i_s.append(df.iloc[row, col])

            # print(n_g, n_i)
            num_g[finger_type] = n_g
            num_i[finger_type] = n_i

            g_s = np.array(g_s)
            i_s = np.array(i_s)
            g_s.sort()
            i_s.sort()

            min_all, max_all = min(g_s.min(), i_s.min()), max(g_s.max(), i_s.max())

            g_s_norm = (g_s - min_all) / (max_all - min_all)
            i_s_norm = (i_s - min_all) / (max_all - min_all)

            # check
            if n_g < NumOfGen:
                one = np.ones(int(NumOfGen) - n_g)
                g_s_norm = np.append(g_s_norm, one)

            if n_i < NumOfImp:
                one = np.ones(int(NumOfImp) - n_i)
                i_s_norm = np.append(i_s_norm, one)

            #             y1 , y0 = np.ones(n_g), np.zeros(n_i)
            y1, y0 = np.ones(int(NumOfGen)), np.zeros(int(NumOfImp))

            assert len(g_s_norm) == len(y1)
            assert len(i_s_norm) == len(y0)
            y[finger_type] = np.concatenate((y1, y0), axis=0)
            scores[finger_type] = np.concatenate((g_s_norm, i_s_norm), axis=0)
        #             break

        return y, scores, num_g, num_i

    def compute_roc(self):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        eer = dict()

        for finger_type in self.classes:
            #             finger_type = '01'
            fpr[finger_type], tpr[finger_type], threshold = roc_curve(self.y[finger_type], 1 - self.scores[finger_type],
                                                                      pos_label=1)
            fnr = 1 - tpr[finger_type]
            eer_threshold = threshold[np.nanargmin(np.absolute(fnr - fpr[finger_type]))]
            eer[finger_type] = np.mean([fpr[finger_type][np.nanargmin(np.absolute((fnr - fpr[finger_type])))],
                                        fnr[np.nanargmin(np.absolute((fnr - fpr[finger_type])))]])
            roc_auc[finger_type] = roc_auc_score(self.y[finger_type], self.scores[finger_type])
        #             break

        return fpr, tpr, eer, roc_auc

# biometrics = ['knuckle', 'fingerprint', 'dynamic_fusion', 'static_fusion', 'min_fusion', 'product_fusion']
biometrics = ['dynamic_fusion', 'static_fusion', 'min_fusion', 'product_fusion', 'tanh_fusion']
csv2roc = dict()
for biometric in tqdm(biometrics):
    print(biometric)
    csv2roc[biometric] = CSV2ROC(fusion_info[biometric])
classes = csv2roc[biometric].classes

labels = dict()
labels['knuckle'] = 'Finger-Knuckle'
labels['fingerprint'] = 'Fingerprint'
labels['dynamic_fusion'] = 'Dynamic Fusion'
labels['static_fusion'] = 'Static (Sum)'
labels['min_fusion'] = 'Static (Product)'
labels['product_fusion'] = 'Static (Min)'

finger_type = '01'
labels = dict()
labels['knuckle'] = 'Finger-Knuckle'
labels['fingerprint'] = 'Fingerprint'
labels['dynamic_fusion'] = 'Combined'

biometrics = ['knuckle', 'fingerprint', 'dynamic_fusion']
colors = cycle(['black', 'red', 'darkblue'])
linestyles = cycle(['-', '--', '-.', ':'])
plt.figure()
lw = 4

for biometric, color, linestyle in zip(biometrics, colors, linestyles):
    plt.plot(csv2roc[biometric].fpr[finger_type], csv2roc[biometric].tpr[finger_type], lw=lw,
             label='{0} (EER = {1:0.5f})'.format(labels[biometric], csv2roc[biometric].eer[finger_type]),
             color=color, linestyle=linestyle)
plt.xticks(fontsize=25, fontweight='bold')
plt.yticks(np.arange(0.7, 1.01, 0.05), fontsize=25, fontweight='bold')
plt.xlim([1e-4, 1.0])
plt.xscale('log')
plt.ylim([0.7, 1.0])
plt.xlabel('FAR', fontsize=25, fontweight='bold')
plt.ylabel('GAR', fontsize=25, fontweight='bold')
# plt.title('Comparative ROC of Finger-Knuckle, Fingerprint and Dynamic Fusion of Finger' + finger_type, fontsize=25, fontweight='bold')
plt.title('Comparative ROC of Fingerprint and Finger-Knuckle', fontsize=25, fontweight='bold')
plt.legend(loc="lower right", prop={'size': 30, 'weight': 'bold'})
plt.grid(linestyle='--')
# plt.savefig("ROC_fp_kn_dynamic_finger_" + finger_type + ".png")
plt.savefig("ROC_fp_kn_dynamic_finger_" + finger_type + ".svg", dpi=150)
plt.savefig("ROC_fp_kn_dynamic_finger_" + finger_type + ".tiff")
plt.show()
