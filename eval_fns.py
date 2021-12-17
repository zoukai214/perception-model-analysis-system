import yaml
import time
import numpy as np
from functools import reduce
import copy

with open('./configs/config.yml') as file:
        configs_global = yaml.load(file, Loader = yaml.FullLoader)

def gen_dict_class_idx(configs):
    # all_class = configs['CLASS_NAME']
    all_class = copy.deepcopy(configs['CLASS_NAME'])

    merge_class = configs['MERGE_CLASS']
    if merge_class is not None:
        for k,v in merge_class.items():
            all_class.append(k) 
    class_idx_to_name = {}
    class_name_to_idx = {}

    for i, name in enumerate(all_class):
            class_idx_to_name[i] = name
            class_name_to_idx[name] = i
    return class_idx_to_name, class_name_to_idx

def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]

def iou_box(gt_boxes,dt_boxes):
    """
    Args:
        gt_boxes,dt_boxes:[num_box,4] array

    """
    num_gt = gt_boxes.shape[0]
    num_dt = dt_boxes.shape[0]

    gt_inds = []
    dt_inds = []
    iou = np.zeros((num_gt,num_dt),dtype=np.float32)

    # for i in range(num_gt):
    #     gt_inds.append(np.where(gt_boxes[i,:]!=False)[0])

    # for j in range(num_dt):
    #     dt_inds.append(np.where(dt_boxes[j,:]!=False)[0])
    
    for k in range(num_gt):
        for l in range(num_dt):
            one_gt_inds = gt_boxes[k]
            one_dt_inds = dt_boxes[l]

             # get overlapping area
            x1 = max(one_gt_inds[0],one_dt_inds[0])
            y1 = max(one_gt_inds[1],one_dt_inds[1])
            x2 = min(one_gt_inds[2],one_dt_inds[2])
            y2 = min(one_gt_inds[3],one_dt_inds[3])
            # compute width and height of overlapping area
            w = x2-x1
            h = y2-y1
     
            if(w<=0 or h<=0):
                iou[k,l]=0
            else:
                intersect = w*h
                gt_area = (one_gt_inds[2]-one_gt_inds[0])*(one_gt_inds[3]-one_gt_inds[1])
                dt_area = (one_dt_inds[2]-one_dt_inds[0])*(one_dt_inds[3]-one_dt_inds[1])
                union = intersect/(gt_area+dt_area-intersect)
                iou[k,l] = union
    return iou


#calculate overlaps for each pair of gt and dt
# input gt_annos, dt_annos, pcd_paths for point iou
# return overlaps: list. len(overlaps) = len(gt_annos)
def calculate_iou(gt_annos,dt_annos,iou_opt,num_parts=1):
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["cls"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["cls"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
       
        gt_num_list = np.array([len(a['cls']) for a in gt_annos_part])
        dt_num_list = np.array([len(a['cls']) for a in dt_annos_part])
        gt_num_part = np.sum(gt_num_list)
        dt_num_part = np.sum(dt_num_list)

        overlap_part = np.zeros((gt_num_part,dt_num_part), dtype = np.float32)
        id_gt = 0
        id_dt = 0

        #读取每一帧中的gt与dt
        for i in range(num_part):
            bbox_coor = gt_annos_part[i]["bbox"]
            gt_boxes = np.concatenate([bbox_coor], axis=1)
            bbox_coor = dt_annos_part[i]["bbox"]
            dt_boxes = np.concatenate([bbox_coor], axis=1)

            iou = iou_box(gt_boxes,dt_boxes)

            num_gt_one_image = gt_boxes.shape[0]
            num_dt_one_image = dt_boxes.shape[0]

            overlap_part[id_gt:id_gt+num_gt_one_image,id_dt:id_dt+num_dt_one_image]=iou

            id_gt+=num_gt_one_image 
            id_dt+=num_dt_one_image
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0    
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps,parted_overlaps,total_gt_num,total_dt_num

def filter_gt_annos(gt_annos,configs):
    eval_by_class = configs['CLASS_EVAL']
    eval_by_difficulty = configs['DIFFICULTY_EVAL']
    
    ## FILTER CLASS
    class_names = configs['CLASS_NAME']
    class_idx_to_name = {}
    class_name_to_idx = {}
    current_classe_int = []
    curr_class = []

    if(eval_by_class):   
        merge_class = configs['MERGE_CLASS']
        curr_class = configs['CURRENT_CLASS']
        curr_class_name = []
    
        class_idx_to_name, class_name_to_idx = gen_dict_class_idx(configs)
 
        for i, name in enumerate(curr_class):
                current_classe_int.append(class_name_to_idx[name])
                curr_class_name.append(class_idx_to_name[class_name_to_idx[name]])
     
        num_class = len(current_classe_int)

        for gt_anno in gt_annos:
            gt_filtered_index = []
            for cls_int in current_classe_int:
                    mask = (np.where(gt_anno['cls']==cls_int))[0]
                    
                    gt_filtered_index.append(mask)

            # print(len(gt_filtered_index))
            if merge_class is not None:
                    for name, subclasses in merge_class.items():
                            masks_subclasses = []
                            for i, name_subclass in enumerate(subclasses):
                                
                                masks_subclasses.append(np.where(gt_anno['cls']==class_name_to_idx[name_subclass])[0])
                            mask = reduce(np.union1d, masks_subclasses)
                            gt_filtered_index.append(mask)
            gt_anno['class_filter'] = gt_filtered_index
    else:
        # consider all categories as one category
        for gt_anno in gt_annos:
                gt_filtered_index = []
                gt_filtered_index.append(np.int32(gt_anno['idx'].reshape(-1,)))
                gt_anno['class_filter'] = gt_filtered_index

    ## FILTER DIFFICULTY
    if(eval_by_difficulty):
        curr_diff = configs['DIFFICULTY']

        for gt_anno in gt_annos:
            gt_filtered_index =[]   
            for diff in curr_diff:
                mask = (np.where(gt_anno['difficulty']==diff))[0]
                gt_filtered_index.append(mask)
            gt_anno['diff_filter'] = gt_filtered_index
    else:
        for gt_anno in gt_annos:
                gt_filtered_index =[]
                gt_filtered_index.append(np.int32(gt_anno['idx'].reshape(-1,)))
                gt_anno['diff_filter'] = gt_filtered_index
    
    return gt_annos

# according to config file,add filter to dt
def filter_dt_annos(dt_annos,configs):
    eval_by_class = configs['CLASS_EVAL']
    eval_by_difficulty = configs['DIFFICULTY_EVAL']

    ## FILTER CLASS
    class_names = configs['CLASS_NAME']
    class_idx_to_name = {}
    class_name_to_idx = {}
    current_classe_int = []
    curr_class = []
    if(eval_by_class):
            merge_class = configs['MERGE_CLASS']
            curr_class = configs['CURRENT_CLASS']
            curr_class_name = []
            class_idx_to_name, class_name_to_idx = gen_dict_class_idx(configs)
     
            for i, name in enumerate(curr_class):
                    current_classe_int.append(class_name_to_idx[name])
                    curr_class_name.append(class_idx_to_name[class_name_to_idx[name]])
            
            num_class = len(current_classe_int)

            for dt_anno in dt_annos:
                    dt_filtered_index = []
                    for cls_int in current_classe_int:
                            mask = (np.where(dt_anno['cls']==cls_int))[0]
                            dt_filtered_index.append(mask)
                    if merge_class is not None:
                            for name, subclasses in merge_class.items():
                                    masks_subclasses = []
                                    for i, name_subclass in enumerate(subclasses):
                                            masks_subclasses.append(np.where(dt_anno['cls']==class_name_to_idx[name_subclass])[0])
                                    mask = reduce(np.union1d, masks_subclasses)
                                    dt_filtered_index.append(mask)
                    dt_anno['class_filter'] = dt_filtered_index
    else:
            # filter the same way as gt
            for dt_anno in dt_annos:
                    dt_filtered_index = []
                    dt_filtered_index.append(np.int32(dt_anno['idx']).reshape(-1,))
                    dt_anno['class_filter'] = dt_filtered_index
    return dt_annos

# calculate # of gt and dt
def calculate_gt_dt_num(gt_annos, dt_annos, curr_cls, curr_diff):
    total_gt, total_dt = 0,0
    for m in range(len(gt_annos)):
        gt_anno = gt_annos[m]
        dt_anno = dt_annos[m]
        indices_gt_cls = gt_anno['class_filter'][curr_cls]
        indices_gt_diff = gt_anno['diff_filter'][curr_diff]
        indices_gt = reduce(np.intersect1d, (indices_gt_cls, indices_gt_diff))
        
        indices_dt_cls = dt_anno['class_filter'][curr_cls]
        indices_dt = indices_dt_cls

        total_gt += len(indices_gt)
        total_dt += len(indices_dt)
    return total_gt,total_dt
# according to thresholds got, samplling num_sample_pts threholds
def get_thresholdsv2(scores, num_gt, num_sample_pts=30, marker_score=0.5):
        scores.sort()
        size = len(scores)

        thresholds=[marker_score]
        if num_sample_pts<2:
                # print('Only one sample threshold point is set')
                return thresholds

        if size<1:
                # print('No tp is found')
                return thresholds

        dist = size//num_sample_pts

        for i in range(num_sample_pts):
                score = scores[i*dist]
                thresholds.append(scores[i*dist])

        return thresholds
# according each gt and dt, find thresholds for pr curve. one tp --> one thresholds based on iou
def get_pr_thresholds_diff(gt_anno, dt_anno, overlaps, curr_cls, curr_diff, min_overlaps):

        indices_gt_cls = gt_anno['class_filter'][curr_cls]      
        indices_gt_diff = gt_anno['diff_filter'][curr_diff]
        indices_gt = reduce(np.intersect1d, ( indices_gt_cls, indices_gt_diff))

        num_valid_gt = len(indices_gt)
        
       
        indices_dt_cls = dt_anno['class_filter'][curr_cls]
        indices_dt = indices_dt_cls

        thresholds = np.zeros((len(gt_anno['cls']), ))

        # print(gt_anno['path'], dt_anno['path'])
        # print(indices_gt, indices_dt)
        if(len(indices_gt)==0):
                return np.zeros([0]), num_valid_gt

        if(len(indices_dt)==0):
                return np.zeros([0]), num_valid_gt

        det_size = len(dt_anno['cls'])
        assigned_detection = [False] * det_size

        tp, fp, fn = 0, 0, 0
        thresh_idx=0


        for index_gt in indices_gt:
            # gt_anno_filtered = gt_anno[index_gt]

            gt_cls = gt_anno['cls'][index_gt,0]
            min_overlap = min_overlaps[gt_cls]
            # print min_overlap

            det_idx = -1
            max_score=-10000
            dt_scores = dt_anno['score']
            
            for index_dt in indices_dt:
                if(assigned_detection[index_dt]):
                        continue

                dt_score = dt_scores[index_dt][0]
                overlap = overlaps[index_gt,index_dt]

                if(dt_score > max_score and overlap>min_overlap):

                        det_idx = index_dt
                        max_score = dt_score
                        
                    
            if(max_score > -10000):
                tp+=1
                thresholds[thresh_idx] = dt_scores[det_idx]
                
                thresh_idx+=1
                assigned_detection[det_idx]=True
        # print(tp)
        return thresholds[:thresh_idx], num_valid_gt

# according to thresholds got, samplling num_sample_pts threholds
def get_thresholds(scores, num_gt, num_sample_pts=30, marker_score=0.5):
        scores.sort()
        size = len(scores)

        thresholds=[marker_score]
        if num_sample_pts<2:
                # print('Only one sample threshold point is set')
                return thresholds

        if size<1:
                # print('No tp is found')
                return thresholds

        dist = size//num_sample_pts
        for i in range(num_sample_pts):
                score = scores[i*dist]
                thresholds.append(scores[i*dist])

        return thresholds

##according to thresh find tp,fp,fn
def compute_statictics(gt_anno,dt_anno,overlaps,curr_cls,curr_diff,thresh,min_overlaps):
    
    ## for corner cases
    indices_gt_cls = gt_anno['class_filter'][curr_cls]
    indices_gt_diff = gt_anno['diff_filter'][curr_diff]
    indices_gt = reduce(np.intersect1d, ( indices_gt_cls, indices_gt_diff))

    indices_dt_cls = dt_anno['class_filter'][curr_cls]
    indices_dt = indices_dt_cls
    fp_ign_ids = []
    tp, fp, fn, fp_ign = 0, 0, 0, 0    

    if(len(indices_gt)==0 and len(indices_dt) !=0):
        fp = np.sum(np.array(dt_anno['score'][indices_dt]>thresh, dtype=np.int32))
        fp_ign = fp.copy()
        return tp, fp, fn, fp_ign

    if(len(indices_gt) !=0 and len(indices_dt) ==0):
            fn = len(indices_gt)
            return tp, fp, fn, fp_ign


    if(len(indices_gt)==0 and len(indices_dt) ==0):
            return tp, fp, fn, fp_ign

    delta_theta = []
    delta_bbox = []
    valid_detection = -1

    assigned_detection = [False] * len(dt_anno['cls'])
    for index_gt in indices_gt:
        gt_cls = gt_anno['cls'][index_gt,0]
        min_overlap = min_overlaps[gt_cls]

        det_idx=-1
        max_overlap=0
        valid_detection=-1

        for index_dt in indices_dt:
            ## finde tp
            dt_score = dt_anno['score'][index_dt]
            overlap = overlaps[index_gt, index_dt]

            if dt_score < thresh:
                    continue
    
            if (assigned_detection[index_dt]):
                    continue
            
            if (overlap > min_overlap and valid_detection == -1):
                valid_detection = 1
                det_idx = index_dt
                max_overlap = overlap
            elif (overlap > min_overlap and valid_detection ==1):
                if (overlap > max_overlap):
                    max_overlap = overlap
                    det_idx = index_dt
        
        if(valid_detection == -1):
            fn+=1
        elif(valid_detection ==1):
            # print('index:', index_gt, det_idx, 'overlap: ',max_overlap, 'score: ',dt_anno['score'][det_idx])
            tp+=1
            assigned_detection[det_idx] = True
    for id_dt in indices_dt:
        dt_score = dt_anno['score'][id_dt]
        ignore_thresh = dt_score<thresh
        if(not(assigned_detection[id_dt] or ignore_thresh)):
            fp+=1
    return tp,fp,fn,fp_ign        

#The main evaluation function
#input: gt_annos,dt_annos,overlaps,configs
#output: et/ot precision recall .etc
def do_eval(gt_annos,dt_annos,overlaps,configs):
    assert len(gt_annos) == len(dt_annos)
    num_frames = len(gt_annos)
    eval_by_class = configs['CLASS_EVAL']
    eval_by_difficulty = configs['DIFFICULTY_EVAL']
    
    min_overlaps = configs['MIN_IOU_THRESH']
    
    class_names = configs['CLASS_NAME']
    class_idx_to_name = {0:'object'}
    class_name_to_idx = {'object':0}
    current_classe_int = [0]

    if(eval_by_class):
        class_idx_to_name, class_name_to_idx = gen_dict_class_idx(configs)

        curr_class = configs['CURRENT_CLASS']
        merge_class = configs['MERGE_CLASS']

        curr_class_name = []
        current_classe_int = []
        for i, name in enumerate(curr_class):
                current_classe_int.append(class_name_to_idx[name])
                curr_class_name.append(class_idx_to_name[class_name_to_idx[name]])
        
        if merge_class is not None:
                for name, subclasses in merge_class.items():
                        current_classe_int.append(class_name_to_idx[name])
                        curr_class_name.append(class_idx_to_name[class_name_to_idx[name]])
        
        num_class = len(current_classe_int)    
    # print(current_classe_int)    
    curr_diffs = [0]
    if(eval_by_difficulty):
            curr_diffs = configs['DIFFICULTY']
    num_class = len(current_classe_int)
    num_difficulty = len(curr_diffs) 
    marker_scores = configs['MARKER_THRESH']

    N_SAMPLE_PTS = configs['NUM_TREHSH_SAMPLE']
    marker_scores = configs['MARKER_THRESH']

    ### basic evaluation
    threshes = np.zeros([num_class,num_difficulty,1,N_SAMPLE_PTS])
    all_statistics = np.zeros([num_class, num_difficulty, 3,N_SAMPLE_PTS])
    total_num_gt = np.zeros([num_class,num_difficulty,1,1])
    total_num_dt = np.zeros([num_class,num_difficulty,1,1])
    

    for i,curr_cls in enumerate(current_classe_int):
        curr_cls_name = class_idx_to_name[curr_cls] #car Pedestrain .eta
        for j,curr_diff in enumerate(curr_diffs): #[0,1,2]
            ## statistic gt, dt nums
            num_gt, num_dt = calculate_gt_dt_num(gt_annos, dt_annos, i, j)
            total_num_gt[i,j,0,0]=num_gt
            total_num_dt[i,j,0,0]=num_dt

            ## statistic pr/aos curve
            print('class:', class_idx_to_name[curr_cls], ' difficulty:', curr_diff)
            thresholdss = []
            total_num_valid_gt = 0
            for l in range(len(gt_annos)):
                thresholds, num_valid_gt = get_pr_thresholds_diff(gt_annos[l],dt_annos[l],overlaps[l],i,j,min_overlaps)
                thresholdss += thresholds.tolist()
                total_num_valid_gt+=num_valid_gt               
            
            thresholdss = np.array(thresholdss)
            thresholdss = np.unique(thresholdss)
            thresholds = get_thresholdsv2(thresholdss, total_num_valid_gt, num_sample_pts=N_SAMPLE_PTS-1, marker_score=marker_scores[i])
            thresholds =  np.array(thresholds)
            print(thresholds)
            for t,thresh in enumerate(thresholds):
                total_tp,total_fp,total_fn,total_fp_ignore = 0,0,0,0

                #     record score 0.5 index
                for m in range(len(gt_annos)):
                    tp,fp,fn,fp_jgnore = compute_statictics(gt_annos[m],dt_annos[m],overlaps[m],i,j,thresh,min_overlaps)
                    total_tp+=tp
                    total_fp+=fp
                    total_fn+=fn
                    total_fp_ignore+=fp_jgnore
                prec_base = total_tp+total_fp
                rec_base = total_tp+total_fn
                if prec_base<=0:
                    prec_base = 1
                if rec_base<=0:
                    rec_base = 1
                # print(thresh, total_tp, total_fp, total_fn, float(total_tp)/float(prec_base), float(total_tp)/float(rec_base))

                all_statistics[i,j,0,t]+=total_tp
                all_statistics[i,j,1,t]+=total_fp
                all_statistics[i,j,2,t]+=total_fn
            for m in range(len(thresholds)):
                threshes[i,j,0,m] = thresholds[m]


    return threshes,total_num_gt,total_num_dt,all_statistics

                       