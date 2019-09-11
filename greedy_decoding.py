import config 
import numpy as np 


skeleton_graph = {i:[] for i in range(config.NUM_KP)}
for i in range(config.NUM_KP):
    for j in range(config.NUM_KP):
        if (i,j) in config.EDGES or (j,i) in config.EDGES:
            skeleton_graph[i].append(j)
            skeleton_graph[j].append(i)
dir_edges = config.EDGES + [edge[::-1] for edge in config.EDGES]

def iterative_bfs(graph, start, path=[]):
    r'''iterative breadth first search from start'''
    q=[(None,start)]
    visited = []
    while q:
	    v=q.pop(0)
	    if not v[1] in visited:
            visited.append(v[1])
            path=path+[v]
            q=q+[(v[1], w) for w in graph[v[1]]]
    return path
    
def connect_adjacent_keypoint(this_skel, current_keypoint, mid_offsets, candidate_keypoints, edge):


    mid_idx = dir_edges.index(edge)
    offsets = mid_offsets[:,:,2*mid_idx:2*mid_idx+2]
    from_kp = tuple(np.round(current_keypoint[edge[0],:2]).astype('int32'))
    proposal = this_skel[edge[0],:2] + offsets[from_kp[1], from_kp[0], :]
    matches = [(i, candidate_keypoints[i]) for i in range(len(candidate_keypoints)) if candidate_keypoints[i]['id'] == edge[1]] 
    matches = [match for match in matches if np.linalg.norm(proposal-match[1]['xy']) <= 32]

    if len(matches) == 0:
	    return None, None

    matches.sort(key=lambda m: np.linalg.norm(m[1]['xy']-proposal))
    to_kp = np.round(matches[0][1]['xy']).astype('int32')

    to_kp_conf = matches[0][1]['conf']

    candidate_keypoints.pop(matches[0][0])
    
    return to_kp,to_kp_conf
    
def group_skeletons(keypoints, mid_offsets, skeleton_graph):
    keypoints.sort(key=(lambda kp: kp['conf']), reverse=True)
  
    skeletons = []
    
    while len(keypoints) > 0:
        kp = keypoints.pop(0)
		
        if any([np.linalg.norm(kp['xy']-s[kp['id'], :2]) <= 10 for s in skeletons]):
            continue
            
        this_skel = np.zeros((config.NUM_KP, 3))
        this_skel[kp['id'], :2] = kp['xy']
        this_skel[kp['id'], 2] = kp['conf']
		   
        path = iterative_bfs(skeleton_graph, kp['id'])[1:] 
        
        for edge in path:
            if this_skel[edge[0],2] == 0:
                continue
		 
            to_kp,to_kp_conf = connect_adjacent_keypoint(this_skel[edge[0],:2], mid_offsets, edge, keypoints)
            if to_kp!=None:
                this_skel[edge[1],:2] = to_kp
                this_skel[edge[1], 2] = to_kp_conf
            else:
                break
	    
        skeletons.append(this_skel)
    return skeletons
