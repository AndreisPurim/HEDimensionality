import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import gc
import os


def generate_dataset(N, M, vector_size=128, min_val = 0, max_val = 16, noise_std=4):
    """
    Generate a dataset of biometric feature vectors.
    Returns:
        samples_flat (list of tuples): [(sample_id, user_id, vector), ...]
    """
    base_vectors = np.random.randint(min_val, max_val, size=(N, vector_size))
    samples = []
    sample_id = 0
    for user_id in range(N):
        for _ in range(M):
            noise = np.random.normal(0, noise_std, vector_size)
            vec = base_vectors[user_id] + noise
            vec = np.clip(np.round(vec), min_val, max_val-1).astype(int)
            samples.append((sample_id, user_id, vec))
            sample_id += 1
    return samples


def dtw_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the DTW distance between two 1D arrays v1 and v2.
    This runs in O(n^2) time and space.
    """
    n = len(v1)
    # Create (n+1)x(n+1) cost matrix filled with +inf
    dtw = np.full((n+1, n+1), np.inf)
    dtw[0, 0] = 0.0

    # Fill the matrix
    for i in range(1, n+1):
        for j in range(1, n+1):
            cost = abs(v1[i-1] - v2[j-1])
            # take last min of insertion, deletion, match
            last_min = min(dtw[i-1, j],    # insertion
                           dtw[i,   j-1],  # deletion
                           dtw[i-1, j-1])  # match
            dtw[i, j] = cost + last_min

    return dtw[n, n]


def squared_euclidean(v1, v2):
    """Compute squared Euclidean distance between two integer vectors."""
    return int(np.sum((v1 - v2)**2))

def compute_match_scores(samples):
    """
    Compute a DataFrame of match scores for every sample pair.
    Columns: sample_id_a, sample_id_b, same_user, score
    """
    records = []
    n = len(samples)
    for i in range(n):
        id_a, user_a, vec_a = samples[i]
        for j in range(n):
            id_b, user_b, vec_b = samples[j]
            # Peba originally stores the results as 24 bits, here we will convert to 16 since numpy doesnt have it
            vec_a = vec_a.astype(np.uint16)
            vec_b = vec_b.astype(np.uint16)
            score = squared_euclidean(vec_a, vec_b)
            score = int(np.sum((vec_a - vec_b)**2))
            if len(vec_a) == 2 and np.array_equal(vec_a, np.array([47, 197])):
                if(np.array_equal(vec_b, np.array([0, 204]))):
                    print("now: ", score)
                    print(vec_a, vec_b, score)
                    v1 = np.array([47, 197])
                    v2 = np.array([0, 204])
                    print(int(np.sum((v1 - v2)**2)))
                    print(list(vec_a), list(vec_b))
                    print(int(np.sum((vec_a - vec_b)**2)))
            same_user = (user_a == user_b)
            records.append({
                'sample_id_a': f"{user_a}_{id_a}",
                'sample_id_b': f"{user_b}_{id_b}",
                'same_user': same_user,
                'score': score
            })
    return pd.DataFrame.from_records(records)

def compute_tar_far_frr_eer(df, n_gen, n_imp, thresholds=None, step=None):
    """
    Given a DataFrame with columns ['same_user', 'score'], compute for each threshold:
      - FAR = false_accepts / total_impostor
      - FRR = false_rejects / total_genuine
    Then find the EER by locating where FAR and FRR cross (with linear interpolation).
    
    Args:
        df: pandas DataFrame with boolean 'same_user' and numeric 'score'.
        thresholds: array‐like of thresholds to test. If None:
            - if step is provided, uses np.arange(0, max_score+1, step)
            - else uses np.unique(scores)
        step: if provided and thresholds is None, defines step size for threshold grid.
        
    Returns:
        rates: DataFrame with columns ['threshold','far','frr']
        eer: interpolated Equal Error Rate
        eer_threshold: threshold at which EER occurs
    """
    scores = df['score'].values
    genuine = df['same_user'].values
    impostor = ~genuine

    max_score = scores.max()
    if thresholds is None:
        if step is not None:
            thresholds = np.arange(0, max_score+2*step+1, step)
        else:
            thresholds = np.unique(scores)

    # Vectorized count of FA/FR for each threshold
    fa = np.array([(impostor & (scores <= t)).sum() for t in thresholds])
    fr = np.array([(genuine & (scores >  t)).sum() for t in thresholds])

    far = fa / n_imp
    frr = fr / n_gen

    rates = pd.DataFrame({
        'threshold': thresholds,
        'false_acceptance': fa,
        'false_rejection': fr, 
        'far':       far,
        'frr':       frr
    })

    # Compute EER: find where FAR-FRR crosses zero
    diff = rates['far'].values - rates['frr'].values
    # Indices where diff goes from negative→positive or zero
    idx = np.where(diff >= 0)[0]

    if idx.size == 0:
        # never crosses: pick the point of minimal |FAR−FRR|
        i = np.argmin(np.abs(diff))
        eer = (rates.at[i,'far'] + rates.at[i,'frr']) / 2
        eer_thr = rates.at[i,'threshold']
    else:
        i1 = idx[0]
        if i1 == 0:
            eer, eer_thr = rates.at[0,'far'], rates.at[0,'threshold']
        else:
            i0 = i1 - 1
            # interpolate threshold at diff=0
            thr0, thr1 = thresholds[i0], thresholds[i1]
            d0, d1     = diff[i0],       diff[i1]
            eer_thr    = np.interp(0, [d0, d1], [thr0, thr1])
            # interpolate error rate at that point
            eer = np.interp(0, [d0, d1], [far[i0], far[i1]])
    
    return rates, eer, eer_thr

def compress(samples, k, max_val = 16):
    X = np.vstack([vec for (_,_,vec) in samples])   # shape (N*M, 128)
    rp = GaussianRandomProjection(n_components=k)
    Z = rp.fit_transform(X) 
    mn, mx = Z.min(), Z.max()
    scale = float(max_val-1)/(mx-mn)
    Zq = np.round((Z - mn)*scale).astype(np.uint8)

    compacted_samples = []
    for i, (sample_id, user_id, _orig_vec) in enumerate(samples):
        compact_vec = Zq[i]                     # the k‐dim uint8 vector
        compacted_samples.append((sample_id, user_id, compact_vec))
    return compacted_samples

def save_samples_to_txt(samples, filename):
    with open(filename, 'w') as f:
        for sample_id, user_id, vec in samples:
            vec_str = ' '.join(map(str, vec))
            f.write(f"{sample_id} {user_id} {vec_str}\n")

if __name__ == "__main__":
    N, M = 10, 10
    vector_size = 128
    max_val = 256
    noise_std = 30
    print(f"[prepared_data.py] Preparing random dataset of size ({N}, {M}) with values in range (0, {max_val})")
    samples = generate_dataset(N, M, max_val=max_val, noise_std=noise_std)
    compression_range = list(range(vector_size, 0, -16)) + [8, 4, 2, 1]

    results = {
        'x': [],
        'e': []
    }
    print("[prepared_data.py] Compressing and calculating expected EER")
    for compression in compression_range:
        os.makedirs("output/normal/",exist_ok=True)
        proc_samples = compress(samples, k = compression, max_val = max_val)
        save_samples_to_txt(proc_samples, f"output/normal/comp_{compression}_samples.txt")
        gc.collect()
        df_scores = compute_match_scores(proc_samples)
        # Maximum number of true rejections: everyone who is not a sample from n
        # So in other words: if I pick a user n, then it's all samples M from all N-1
        max_negative = (M-1)*M*N
        # maximum number of true positives: everyone who could be true, so that is M*M*N
        max_positive = (N-1)*M*M*N
        rates_df, eer, eer_thr = compute_tar_far_frr_eer(df_scores, max_negative, max_positive, step=20)
        print(f"\tCompression n = {compression} -> EER = {eer:.4f} at threshold ≃ {eer_thr:.1f}")
        df_scores.to_csv(f"output/normal/comp_{compression}_scores.csv", index=False)
        rates_df.to_csv(f"output/normal/comp_{compression}_rates.csv")
        results['x'].append(compression)
        results['e'].append(eer)
    print("[prepared_data.py] Saving to output/normal/results.csv")
    pd.DataFrame(results).to_csv("output/normal/results.csv")

