# OKS-NMS

```python
def OKS_NMS(skeletons_candidates, threshold):
    
    candidates = sorted(skeletons_candidates, key=lambda tmp:tmp['score'], reverse=True)

    i = 0
    while True:

        if i >= len (candidates):
            break

        j = i + 1
        while True:

            if j >=len (candidates):
                break

            kpt1 = candidates[i]['keypoints']
            kpt2 = candidates[j]['keypoints']

            if ComputeOKS(kpt1,kpt2) > threshold:
                candidates.pop(j)

            j += 1

        i += 1
    return  candidates
```
# Greedy Decoding in PersonLab (eccv 2018)

```python
 config.EDGES = [
        (0, 14),
        (0, 13),
        (0, 4),
        (0, 1),
        (14, 16),
        (13, 15),
        (4, 10),
        (1, 7),
        (10, 11),
        (7, 8),
        (11, 12),
        (8, 9),
        (4, 5),
        (1, 2),
        (5, 6),
        (2, 3)
    ]
    
def group_skeletons(keypoints, mid_offsets):
    
    keypoints.sort(key=(lambda kp: kp['conf']), reverse=True)
  
    skeletons = []
    
    dir_edges = config.EDGES + [edge[::-1] for edge in config.EDGES]
    
    skeleton_graph = {i:[] for i in range(config.NUM_KP)}
    for i in range(config.NUM_KP):
        for j in range(config.NUM_KP):
            if (i,j) in config.EDGES or (j,i) in config.EDGES:
                skeleton_graph[i].append(j)
                skeleton_graph[j].append(i)
    
    while len(keypoints) > 0:
    	
        kp = keypoints.pop(0)
		
        if any([np.linalg.norm(kp['xy']-s[kp['id'], :2]) <= 10 for s in skeletons]):
            continue
            
        this_skel = np.zeros((config.NUM_KP, 3))
        this_skel[kp['id'], :2] = kp['xy']
        this_skel[kp['id'], 2] = kp['conf']
 
		def iterative_bfs(graph, start, path=[]):
    		'''iterative breadth first search from start'''
    		q=[(None,start)]
    		
		    visited = []
		    while q:
		       
		        v=q.pop(0)
		        
		        if not v[1] in visited:
		            visited.append(v[1])
		            path=path+[v]
		            q=q+[(v[1], w) for w in graph[v[1]]]
		        return path
		   
        path = iterative_bfs(skeleton_graph, kp['id'])[1:] 
        
        for edge in path:
            if this_skel[edge[0],2] == 0:
                continue
            
            mid_idx = dir_edges.index(edge)
            
            offsets = mid_offsets[:,:,2*mid_idx:2*mid_idx+2]
            from_kp = tuple(np.round(this_skel[edge[0],:2]).astype('int32'))
            proposal = this_skel[edge[0],:2] + offsets[from_kp[1], from_kp[0], :]
			      matches = [(i, keypoints[i]) for i in range(len(keypoints)) if keypoints[i]['id'] == edge[1]] 
			      matches = [match for match in matches if np.linalg.norm(proposal-match[1]['xy']) <= 32]
           
            if len(matches) == 0:
                continue
            
            matches.sort(key=lambda m: np.linalg.norm(m[1]['xy']-proposal)
            to_kp = np.round(matches[0][1]['xy']).astype('int32')
     
            to_kp_conf = matches[0][1]['conf']
            
            keypoints.pop(matches[0][0])
            this_skel[edge[1],:2] = to_kp
            this_skel[edge[1], 2] = to_kp_conf
        skeletons.append(this_skel)

    return skeletons

```
