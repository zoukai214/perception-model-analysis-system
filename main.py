#coding:utf-8
import os
import sys
import pathlib
import csv
import yaml
import numpy as np
import time
import matplotlib.pyplot as plt


import eval_fns


with open('./configs/config.yml') as file:
        configs = yaml.load(file, Loader = yaml.FullLoader)

config_path = configs['config_path']
for path in config_path:
        if path not in sys.path:
                sys.path.insert(0,path)

diff_idx_to_name = configs['diff_idx_to_name']
class_idx_to_name, class_name_to_idx = eval_fns.gen_dict_class_idx(configs)
print(class_idx_to_name)

def read_info(eval_list_path,root,info_type,DEBUG,configs):

    with open(str(eval_list_path),'r') as f:
        eval_list = f.readlines()
    # if DEBUG is True:
    #     eval_list = eval_list[:configs['DEBUG_FRAME_NUM']]
    infos = []
    for i ,path in enumerate(eval_list):
        path=str(path).replace('\n','')
        info_path = os.path.join(root,path)
        info = {}
        info['image_id'] = i
        info['image_path'] = path
        annos = read_anno(info_path,info_type,configs)

        info['annos'] = annos
        infos.append(info)
        if len(annos)<1:
            print("empty file: ",path)
            exit(1)
            
    return infos

def read_anno(info_path,info_type,configs):
    with open(str(info_path),'r',encoding='utf-8') as f:
        lines = f.readlines()
    annotation_contents = []
    for line in lines:
            contents= line.replace('\n','')
            contents= contents.split(' ')
            annotation_contents.append(contents)

    annos = {}
    annos['path'] = info_path
    annos['cls'] = np.array([class_name_to_idx[x[0]] for x in annotation_contents]).reshape(-1,1)
    annos['idx']=np.array(np.arange(len(annos['cls'])),dtype=np.float32).reshape(-1,1)

    # annos['truncated'] = np.array([float(x[1]) for x in annotation_contents]).reshape(-1,1)
    # annos['occluded'] = np.array([int(x[2]) for x in annotation_contents]).reshape(-1,1)

    annos['alpha'] = np.array([float(x[3]) for x in annotation_contents]).reshape(-1,1)
    annos['bbox'] = np.array([[float(elem) for elem in x[4:8]] for x in annotation_contents]).reshape(-1,4)
    if info_type == 'GT':
        occlusion = configs['DIFFICULTY_CONFIG']['occlusion']
        truncation = configs['DIFFICULTY_CONFIG']['truncation']
        height = configs['DIFFICULTY_CONFIG']['height']


        Trunc = []
        Occlu = []
        diff = []
        for  x in annotation_contents:
            gt_truncated = float(x[1])
            gt_occluded = int(x[2])
            gt_height = float(x[7])-float(x[5])

            if((gt_occluded==int(occlusion[0])) and (gt_truncated<=float(truncation[0])) and (gt_height>=height[0])):
                difficulty = 0
            # if(gt_occluded==occlusion[1] or truncation[0]<gt_truncated<= truncation[1] or height[1]<=gt_height<height[0]):
            #     difficulty =1
            elif(gt_occluded>=occlusion[2] or gt_truncated>truncation[2] or gt_height<height[2]):
                difficulty = 2
            else:
                difficulty =1
            # Trunc.append(gt_truncated)
            # Occlu.append(Occlu)
            diff.append(difficulty)

        # annos['truncated'] = np.array(Trunc).reshape(-1,1)
        # annos['occluded'] = np.array(Occlu).reshape(-1,1)
        annos['difficulty'] = np.array(diff).reshape(-1,1)

    if info_type == 'DT':
        annos['score'] = np.array([float(x[15]) for x in annotation_contents]).reshape(-1,1)

    return annos

def check_base(bases):
    n_base = len(bases)
    for i in range(n_base):
            if bases[i] <= 0:
                    bases[i] = 1
    return bases

def save_result(ret,configs):
    result_root = str((pathlib.Path(configs['DT_ANNO']['ROOT_PATH'])))
    if configs['RESULT_ROOT'] is not None:
            result_root = configs['RESULT_ROOT']

    eval_by_class = configs['CLASS_EVAL']
    eval_by_difficulty = configs['DIFFICULTY_EVAL']
  
    num_sample_pt = configs['NUM_TREHSH_SAMPLE']

    num_class = 1
    num_curr_class = 1
    if(eval_by_class):
        curr_class = configs['CURRENT_CLASS'] 
        merge_class = configs['MERGE_CLASS']   
        num_curr_class = len(curr_class)
        num_class = num_curr_class
        if merge_class is not None:
                num_class += len(merge_class)
                curr_class = curr_class + list(merge_class.keys())
    num_diff = 1
    curr_diff_dist = []

    if(eval_by_difficulty):
        curr_diff = configs['DIFFICULTY']
        # curr_diff_dist = configs['DIFFICULTY_DISTANCE']
        num_diff = len(curr_diff)
    threshes, gt_num, dt_num, all_statistics= ret
    #save basic eval result per 

    for i ,diff in enumerate(curr_diff):
        pr_curve = np.zeros([num_diff,num_class,num_sample_pt,3])
        path_result = pathlib.Path(result_root+'/diff/'+str(diff)+'/'+configs['RESULT_FILE'])
        if pathlib.Path.exists(path_result.parent) == False:
            os.makedirs(str(path_result.parent))
        ## detection result 
        with open(str(path_result),'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Class','GT-Cnt','DT-Cnt','Recall','Precise','f1'])
            gtn_obj = np.sum(gt_num[:num_curr_class,i])
            dtn_obj = np.sum(dt_num[:num_curr_class,i])

            total_tp = np.sum(all_statistics[:num_curr_class,i,0,0])
            total_fp = np.sum(all_statistics[:num_curr_class,i,1,0])
            total_fn = np.sum(all_statistics[:num_curr_class,i,2,0])
            
            bases_1 = check_base([total_tp+total_fn,total_tp+total_fp])
            recall = total_tp/bases_1[0]
            precision = total_tp/bases_1[1]
            bases_2 = check_base([precision+recall])

            f1 = 2*(precision*recall)/bases_2[0]
            writer.writerow(['total',gtn_obj, dtn_obj, recall,precision,f1])

            for k in range(num_class):
                cls_name = curr_class[k]
                gtn = np.sum(gt_num[k,i,0,0])
                dtn = np.sum(dt_num[k,i,0,0])

                tp = np.sum(all_statistics[k,i,0,0])
                fp = np.sum(all_statistics[k,i,1,0])
                fn = np.sum(all_statistics[k,i,2,0])
                bases_1 = check_base([tp+fn,tp+fp])

                recall = tp/bases_1[0]
                precision = tp/bases_1[1]
                bases_2 = check_base([precision+recall])
                f1 = 2*(precision*recall)/bases_2[0]
                print("cls:{},tp:{},fp:{}".format(cls_name,tp,fp))
                writer.writerow([cls_name, gtn, dtn, recall, precision, f1])

                ##pr curve
                path_pr = pathlib.Path()
                for t in range(num_sample_pt):
                    tp = np.sum(all_statistics[k,i,0,t])
                    fp = np.sum(all_statistics[k,i,1,t])
                    fn = np.sum(all_statistics[k,i,2,t])

                    bases = check_base([tp + fn, tp + fp])
                    rec = tp/bases[0]
                    prec = tp/bases[1]
                    thresh = threshes[k,i,0,t]
                    pr_curve[i,k,t,0] = thresh
                    pr_curve[i,k,t,1] = prec
                    pr_curve[i,k,t,2] = rec
        ## save pr curve
        # path_pr = pathlib.Path(result_root+'/diff/'+str(diff)+'/')
        # print(path_pr)
        for pr_cls in range(num_class):
            cls_name = curr_class[pr_cls]
            path_pr_txt =pathlib.Path( result_root+'/diff/'+str(diff)+'/'+'pr_'+cls_name+'.txt')
            with open(path_pr_txt, 'w') as f:
                for t in range(num_sample_pt):
                    thresh = pr_curve[i,pr_cls,t,0]
                    prec = pr_curve[i,pr_cls,t,1]
                    rec = pr_curve[i,pr_cls,t,2]
                    f.write(str(thresh) +' '+ str(prec)+' '+str(rec)+'\n')

                y = pr_curve[i,pr_cls,:,1]
                x = pr_curve[i,pr_cls,:,2]

                y = np.append(y,y[-1])
                x = np.append(x,0.0)

                y = y[::-1]
                x = x[::-1]
                f.write('AP: '+ str(np.trapz(y,x=x)))

            plt.figure(figsize=(15,10))
            plt.xlim(right=1, left=0)
            plt.ylim(top=1, bottom=0)
            plt.plot(pr_curve[i,pr_cls,:,2], pr_curve[i,pr_cls,:,1],'bo-')
            for t,x,y in zip(pr_curve[i,pr_cls,:,0], pr_curve[i,pr_cls,:,2],pr_curve[i,pr_cls,:,1]):
                    label = "({:.3f}, {:.3f})".format(x, y)
                    plt.annotate(label, (x,y), textcoords="offset points", xytext=(25,50), ha='center', rotation=45) 
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.savefig(pathlib.Path(result_root+'/diff/'+str(diff)+'/'+cls_name+'.svg'))
            # print("{}".format(str(path_pr)+cls_name+'.svg'))
            plt.clf() 




def evalutionV2(gt_annos,dt_annos,config):
    assert len(gt_annos) == len(dt_annos)

    iou_opt = configs['IOU_OPT']
    print('start calculating iou... it may take some time if you are using point iou')
    rets = eval_fns.calculate_iou(gt_annos,dt_annos,iou_opt)
    overlaps,parted_overlaps,total_gt_num,total_dt_num = rets
    
    print('start filtering gt and dt...')
    gt_annos = eval_fns.filter_gt_annos(gt_annos, configs)
    dt_annos = eval_fns.filter_dt_annos(dt_annos, configs) 

    print('start evaluation...')
    ret  = eval_fns.do_eval(gt_annos,dt_annos,overlaps,configs)
    threshes,gt_num,dt_num,all_statistics = ret

    print('save eval result')
    #write into csv files
   # save_result(ret,configs)

def main():

    Eval_list = configs['EVAL_LIST']
    GT_root = configs['GT_ANNO']['ROOT_PATH']
    DT_root = configs['DT_ANNO']['ROOT_PATH']
    Debug = configs['DEBUG']

    gt_infos = read_info(Eval_list,GT_root,configs['GT_ANNO']['TYPE'],Debug,configs)
    dt_infos = read_info(Eval_list,DT_root,configs['DT_ANNO']['TYPE'],Debug,configs)

    gt_annos = [info["annos"] for info in gt_infos]
    dt_annos = [info["annos"] for info in dt_infos]
    # print(gt_annos[1])
    # print(dt_annos[1])
    evalutionV2(gt_annos,dt_annos,configs)



   


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('time cost: ', (end - start)/60.0)
