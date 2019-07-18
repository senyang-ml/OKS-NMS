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
