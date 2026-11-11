from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist, pdist, euclidean
from sklearn.model_selection import KFold
from kneed import KneeLocator
import time
import os
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize, StandardScaler
import umap
from scipy.signal import savgol_filter
from scipy.special import softmax
from scipy.stats import entropy

SEED = 0
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


random.seed(SEED)
np.random.seed(SEED)


class MIS_Method:
    def __init__(self, data, n_jobs=1, random_state=0):
        self.data = np.asarray(data, dtype=np.float64)
        self.recommended_cluster_number = None
        self.n_jobs = n_jobs
        self.random_state = int(random_state)
        self._manifold_data = None
        self.name = "MIS_Method"
        self.min_n_components = 5
        self.min_n_neighbors = 35
        self.min_dist = 0.5
        self.threshold = 0.8
        # NEW: cache for MED
        self.ps_scores_ = None
        self.k_list_ = None
        self.n_splits = 5

    def get_name(self):
        return self.name

    def _learn_manifold(self, n_neighbors, n_components):
        """
        Learns the data manifold using UMAP and creates a low-dimensional embedding.
        """
        print(f"\nStage 1: Learning data manifold with UMAP...")
        # Scaling is important for UMAP
        scaled_data = StandardScaler().fit_transform(self.data)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            n_components=n_components,
            random_state=self.random_state,
            n_jobs = 1,
            init="spectral",
        )
        self._manifold_data = reducer.fit_transform(scaled_data)
        print("Manifold learning complete.")

    # def _calculate_ps_for_k(self, k, n_splits=5):
    def _calculate_ps_for_k(self, k, n_splits=None):
        if n_splits is None:
            n_splits = getattr(self, "n_splits", 5)
        X = self._manifold_data
        n = X.shape[0]
        if n < 4:
            return 0.0

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        split_scores = []
        for train_idx, test_idx in kf.split(np.arange(n)):
            X_train, X_test = X[train_idx], X[test_idx]

            # for okb
            n_train = X_train.shape[0]
            n_test = X_test.shape[0]

            if k >= n_train or k >= n_test:
                return 0.0

            km_train = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=self.random_state,
                algorithm="lloyd",
            ).fit(X_train)

            km_test = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=self.random_state,
                algorithm="lloyd",
            ).fit(X_test)

            test_labels = km_test.labels_
            pred_labels = km_train.predict(X_test)

            cluster_strengths = []
            for j in range(k):
                test_cluster_j = np.where(test_labels == j)[0]
                if len(test_cluster_j) < 2:
                    continue
                pred_cluster_j = pred_labels[test_cluster_j]
                co_membership_count = np.sum(pred_cluster_j[:, None] == pred_cluster_j) - len(test_cluster_j)
                total_pairs = len(test_cluster_j) * (len(test_cluster_j) - 1)
                if total_pairs > 0:
                    cluster_strengths.append(co_membership_count / total_pairs)

            if cluster_strengths:
                split_scores.append(np.min(cluster_strengths))

        return float(np.mean(split_scores)) if split_scores else 0.0

    def find_optimal_k(self, max_k=None):
        n_samples = self.data.shape[0]
        if max_k is None:
            max_k = int(np.floor(n_samples ** 0.5) / 2)
            max_k = max(2 + 1, max_k)

        k_list = list(range(2, max_k + 1))
        if not k_list:
            self.recommended_cluster_number = 2
            return 2

        n_components = min(self.min_n_components, n_samples // 2, self.data.shape[1])
        n_neighbors = min(self.min_n_neighbors, n_samples - 1)
        self._learn_manifold(n_components=n_components, n_neighbors=n_neighbors)

        ps_scores = [self._calculate_ps_for_k(k) for k in tqdm(k_list, desc="Stage 2: Assessing Stability on Manifold")]

        if not ps_scores:
            raise RuntimeError("Could not compute Prediction Strength scores.")

        self.ps_scores_ = np.asarray(ps_scores, dtype=np.float64)
        self.k_list_ = np.asarray(k_list, dtype=int)

        stable_k = [k_list[i] for i, score in enumerate(ps_scores) if score > self.threshold]

        if stable_k:
            self.recommended_cluster_number = np.max(stable_k)
        else:
            self.recommended_cluster_number = k_list[np.argmax(ps_scores)]

        return self.recommended_cluster_number



class DCS_Method:
    def __init__(self, data, k_list, random_state=0):
        self.data = np.asarray(data, dtype=np.float64)
        self.cluster_list = list(k_list)
        self.p = self.data.shape[1]
        self.random_state = int(random_state)
        self.name = "DCS_Method"

    def get_name(self):
        return self.name

    def _compute_distortion(self, k):
        km = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=self.random_state,
            algorithm="lloyd"
        ).fit(self.data)
        return float(km.inertia_) / float(self.data.shape[0]) / float(self.p)

    def Distortions(self):
        distortions = []
        for k in tqdm(self.cluster_list, desc="Computing Distortions"):
            distortions.append(self._compute_distortion(k))
        self.distortions = np.array(distortions)
        return self.distortions

    def Jumps(self, distortions=None):
        if distortions is None:
            distortions = self.distortions

        log_d = np.log(distortions + 1e-12)

        n = len(log_d)
        if n < 3:
            smoothed = log_d
        else:
            wl = min(15, n if n % 2 == 1 else n - 1)
            wl = max(3, wl)
            poly = min(5, wl - 1)
            smoothed = savgol_filter(log_d, window_length=wl, polyorder=poly, mode="constant")

        curvature = np.gradient(np.gradient(smoothed))

        slope = np.gradient(smoothed)
        stability_score = curvature / (np.abs(slope) + 1e-6)

        best_index = np.argmax(stability_score)
        self.recommended_cluster_number = self.cluster_list[best_index]

        return self.recommended_cluster_number, stability_score



class MED_Method:
    def __init__(self, data, n_jobs=1, random_state=0):
        self.data = np.asarray(data, dtype=np.float64)
        self.name = "MED_Method"
        self.n_jobs = n_jobs
        self.random_state = int(random_state)
        self.recommended_cluster_number = None

    def get_name(self):
        return self.name

    @staticmethod
    def _normalized_entropy_from_scores(scores: np.ndarray) -> float:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size <= 1:
            return 0.0
        p = softmax(scores)
        h = entropy(p)
        h_max = np.log(scores.size)
        return float(h / h_max) if h_max > 0 else 0.0

    def find_optimal_k(self, max_k=None):
        n_samples = self.data.shape[0]
        if max_k is None:
            max_k = int(np.sqrt(n_samples)) + 1

        k_list = np.array(range(2, max_k + 1), dtype=int)
        if k_list.size < 2:
            self.recommended_cluster_number = 2
            return 2

        print("MED Stage 1: Running MIS expert ...")
        mis_expert = MIS_Method(self.data, n_jobs=1, random_state=self.random_state)
        k_mis = int(mis_expert.find_optimal_k(max_k=max_k))

        if getattr(mis_expert, "ps_scores_", None) is None:
            ps_scores = np.asarray([mis_expert._calculate_ps_for_k(int(k)) for k in k_list], dtype=np.float64)
        else:
            ps_scores = np.asarray(mis_expert.ps_scores_, dtype=np.float64)

        p = softmax(ps_scores)
        p_sorted = np.sort(p)[::-1]
        margin = p_sorted[0] - p_sorted[1]
        spread = np.std(p)

        if margin > spread:
            final_k = k_mis
            print("MED Stage 2: Running MIS expert ...")
            print(f"  - MIS is more decisive ({margin:.4f} > {spread:.4f}). Trusting MIS.")
        else:
            H_mis = self._normalized_entropy_from_scores(ps_scores)
            print("ps_scores (MIS):", ps_scores.tolist())
            print(f"  - MIS Prediction: {k_mis}")
            print(f"  - MIS Normalized Entropy: {H_mis:.4f}")

            print("\nMED Stage 2: Running DCS expert ...")
            dcs_expert = DCS_Method(self.data, k_list=k_list, random_state=self.random_state)
            dcs_expert.Distortions()
            k_dcs, stability_score = dcs_expert.Jumps()
            k_dcs = int(k_dcs)

            stability_score = np.asarray(stability_score, dtype=np.float64)
            H_dcs = self._normalized_entropy_from_scores(stability_score)
            print("stability_score (DCS):", stability_score.tolist())
            print(f"  - DCS Prediction: {k_dcs}")
            print(f"  - DCS Normalized Entropy: {H_dcs:.4f}")

            print("\nMED Stage 3: Parameter-free dispatch (lower entropy wins) ...")
            if H_dcs < H_mis:
                print(f"  - DCS is more decisive ({H_dcs:.4f} < {H_mis:.4f}). Trusting DCS.")
                final_k = k_dcs
            else:
                print(f"  - MIS is more decisive ({H_mis:.4f} <= {H_dcs:.4f}). Trusting MIS.")
                final_k = k_mis

        print(f"\nFinal Optimal k from MED: {final_k}")
        self.recommended_cluster_number = int(final_k)
        return int(final_k)

import numpy as np
from scipy.stats import entropy
from scipy.special import softmax

class MED_Method_confidence:
    """
    MED with confidence estimation that does NOT change MED's routing decision.
    """
    name = "MED"

    def __init__(self, data, n_jobs=1, random_state=0, eps=1e-12):
        self.data = np.asarray(data, dtype=np.float64)
        self.n_jobs = n_jobs
        self.random_state = int(random_state)
        self.eps = float(eps)

        self.recommended_cluster_number = None

        # expert outputs
        self.k_mis_ = None
        self.k_dcs_ = None
        self.k_med_ = None

        # score profiles
        self.ps_scores_ = None
        self.dcs_scores_ = None
        self.k_list_ = None

        # entropy / decisiveness
        self.H_mis_ = None
        self.H_dcs_ = None
        self.D_mis_ = None
        self.D_dcs_ = None

        # softmax-derived distributions
        self.p_mis_ = None
        self.p_dcs_ = None
        self.q_mis_ = None
        self.q_dcs_ = None

        # routing diagnostics
        self.selected_expert_ = None
        self.early_accepted_ = False
        self.margin_mis_ = None
        self.spread_mis_ = None
        self.early_margin_signed_ = None
        self.dispatch_margin_signed_ = None

        # confidence
        self.confidence_margin_raw_ = None
        self.confidence_ = None

    def get_name(self):
        return self.name

    @staticmethod
    def _safe_softmax(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size == 0:
            return np.asarray([], dtype=np.float64)
        z = scores - np.max(scores)
        exp_z = np.exp(z)
        denom = np.sum(exp_z)
        if denom <= 0:
            return np.ones_like(scores, dtype=np.float64) / float(len(scores))
        return exp_z / denom

    def _normalized_entropy_from_scores(self, scores: np.ndarray) -> float:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size <= 1:
            return 0.0
        p = self._safe_softmax(scores)
        p = np.clip(p, self.eps, 1.0)
        h = -np.sum(p * np.log(p))
        h_max = np.log(scores.size)
        return float(h / h_max) if h_max > 0 else 0.0

    def _decisiveness_from_entropy(self, H: float) -> float:
        return float(np.clip(1.0 - float(H), 0.0, 1.0))

    def _robust_confidence_distribution(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        if s.size == 0:
            return np.asarray([], dtype=np.float64)

        mu = np.mean(s)
        sd = np.std(s)
        if sd < self.eps:
            return np.ones_like(s, dtype=np.float64) / float(len(s))

        z = (s - mu) / (sd + self.eps)
        r = np.log1p(np.exp(z))
        denom = np.sum(r)
        if denom <= self.eps:
            return np.ones_like(s, dtype=np.float64) / float(len(s))

        q = r / denom
        return q.astype(np.float64)

    def _selected_mass(self, q: np.ndarray, k_list: np.ndarray, k_hat: int) -> float:
        idx = np.where(k_list == int(k_hat))[0]
        if idx.size == 0:
            return 0.0
        return float(q[idx[0]])

    def _neighborhood_mass(self, q: np.ndarray, k_list: np.ndarray, k_hat: int) -> float:
        mask = np.isin(k_list, [int(k_hat) - 1, int(k_hat), int(k_hat) + 1])
        if not np.any(mask):
            return 0.0
        return float(np.sum(q[mask]))

    def _early_accept_confidence(self, q_mis: np.ndarray, k_list: np.ndarray, k_mis: int,
                                 margin_mis: float, spread_mis: float) -> float:
        q_sel = self._selected_mass(q_mis, k_list, k_mis)
        u = (float(margin_mis) - float(spread_mis)) / (float(margin_mis) + float(spread_mis) + self.eps)
        u = float(np.clip(u, 0.0, 1.0))
        c = np.sqrt(max(0.0, u * q_sel))
        return float(np.clip(c, 0.0, 1.0))

    def _dispatch_confidence(self, q_sel: np.ndarray, k_list: np.ndarray, k_hat: int,
                             D_sel: float, D_rej: float) -> float:
        a = (float(D_sel) - float(D_rej)) / (float(D_sel) + float(D_rej) + self.eps)
        a = float(np.clip(a, 0.0, 1.0))

        m_sel = self._neighborhood_mass(q_sel, k_list, k_hat)

        neigh_size = 0
        for kk in [int(k_hat) - 1, int(k_hat), int(k_hat) + 1]:
            if kk in set(k_list.tolist()):
                neigh_size += 1
        u_N = float(neigh_size) / float(len(k_list)) if len(k_list) > 0 else 0.0

        b = (float(m_sel) - u_N) / (1.0 - u_N + self.eps)
        b = float(np.clip(b, 0.0, 1.0))

        c = (2.0 * a * b) / (a + b + self.eps)
        return float(np.clip(c, 0.0, 1.0))

    def find_optimal_k(self, max_k=None):
        n_samples = self.data.shape[0]
        if max_k is None:
            max_k = int(np.sqrt(n_samples)) + 1

        k_list = np.array(range(2, max_k + 1), dtype=int)
        self.k_list_ = k_list.copy()

        if k_list.size < 2:
            self.recommended_cluster_number = 2
            self.k_med_ = 2
            self.confidence_ = 0.5
            return 2

        print("MED Stage 1: Running MIS expert ...")
        mis_expert = MIS_Method(self.data, n_jobs=1, random_state=self.random_state)
        k_mis = int(mis_expert.find_optimal_k(max_k=max_k))
        self.k_mis_ = k_mis

        if getattr(mis_expert, "ps_scores_", None) is None:
            ps_scores = np.asarray([mis_expert._calculate_ps_for_k(int(k)) for k in k_list], dtype=np.float64)
        else:
            ps_scores = np.asarray(mis_expert.ps_scores_, dtype=np.float64)

        self.ps_scores_ = ps_scores.copy()

        p_mis = self._safe_softmax(ps_scores)
        q_mis = self._robust_confidence_distribution(ps_scores)
        self.p_mis_ = p_mis.copy()
        self.q_mis_ = q_mis.copy()

        p_sorted = np.sort(p_mis)[::-1]
        margin_mis = float(p_sorted[0] - p_sorted[1]) if p_sorted.size >= 2 else 0.0
        spread_mis = float(np.std(p_mis))
        self.margin_mis_ = margin_mis
        self.spread_mis_ = spread_mis
        self.early_margin_signed_ = float(margin_mis - spread_mis)

        H_mis = self._normalized_entropy_from_scores(ps_scores)
        D_mis = self._decisiveness_from_entropy(H_mis)
        self.H_mis_ = H_mis
        self.D_mis_ = D_mis

        q_mis_sel = self._selected_mass(q_mis, k_list, k_mis)

        print("ps_scores (MIS):", ps_scores.tolist())
        print(f"  - MIS Prediction: {k_mis}")
        print(f"  - MIS Normalized Entropy: {H_mis:.4f}")
        print(f"  - MIS Decisiveness: {D_mis:.4f}")
        print(f"  - MIS Selected Mass q_mis: {q_mis_sel:.4f}")
        print(f"  - MIS Top-2 Margin: {margin_mis:.4f}")
        print(f"  - MIS Spread: {spread_mis:.4f}")

        if margin_mis > spread_mis:
            self.early_accepted_ = True
            self.selected_expert_ = "MIS"
            self.k_med_ = k_mis
            self.recommended_cluster_number = k_mis

            conf = self._early_accept_confidence(
                q_mis=q_mis,
                k_list=k_list,
                k_mis=k_mis,
                margin_mis=margin_mis,
                spread_mis=spread_mis,
            )
            self.confidence_margin_raw_ = float(margin_mis - spread_mis)
            self.confidence_ = conf

            print("MED Stage 2: Early accept triggered.")
            print(f"  - MIS is more decisive ({margin_mis:.4f} > {spread_mis:.4f}). Trusting MIS.")
            print(f"  - Confidence margin: {self.confidence_margin_raw_:.4f}")
            print(f"  - MED Confidence: {self.confidence_:.4f}")
            print(f"\nFinal Optimal k from MED: {k_mis}")
            return int(k_mis)

        print("\nMED Stage 2: Running DCS expert ...")
        dcs_expert = DCS_Method(self.data, k_list=k_list, random_state=self.random_state)
        dcs_expert.Distortions()
        k_dcs, dcs_scores = dcs_expert.Jumps()
        k_dcs = int(k_dcs)
        dcs_scores = np.asarray(dcs_scores, dtype=np.float64)

        self.k_dcs_ = k_dcs
        self.dcs_scores_ = dcs_scores.copy()

        p_dcs = self._safe_softmax(dcs_scores)
        q_dcs = self._robust_confidence_distribution(dcs_scores)
        self.p_dcs_ = p_dcs.copy()
        self.q_dcs_ = q_dcs.copy()

        H_dcs = self._normalized_entropy_from_scores(dcs_scores)
        D_dcs = self._decisiveness_from_entropy(H_dcs)
        self.H_dcs_ = H_dcs
        self.D_dcs_ = D_dcs

        q_dcs_sel = self._selected_mass(q_dcs, k_list, k_dcs)

        print("dcs_scores (DCS):", dcs_scores.tolist())
        print(f"  - DCS Prediction: {k_dcs}")
        print(f"  - DCS Normalized Entropy: {H_dcs:.4f}")
        print(f"  - DCS Decisiveness: {D_dcs:.4f}")
        print(f"  - DCS Selected Mass q_dcs: {q_dcs_sel:.4f}")

        print("\nMED Stage 3: Original entropy-guided dispatch ...")
        if H_dcs < H_mis:
            self.selected_expert_ = "DCS"
            self.k_med_ = k_dcs
            self.recommended_cluster_number = k_dcs
            self.dispatch_margin_signed_ = float(D_dcs - D_mis)
            self.confidence_margin_raw_ = float(abs(H_dcs - H_mis))

            conf = self._dispatch_confidence(
                q_sel=q_dcs,
                k_list=k_list,
                k_hat=k_dcs,
                D_sel=D_dcs,
                D_rej=D_mis,
            )
            self.confidence_ = conf

            print(f"  - DCS is more decisive ({H_dcs:.4f} < {H_mis:.4f}). Trusting DCS.")
            print(f"  - Confidence margin: {self.confidence_margin_raw_:.4f}")
            print(f"  - MED Confidence: {self.confidence_:.4f}")
            print(f"\nFinal Optimal k from MED: {k_dcs}")
            return int(k_dcs)
        else:
            self.selected_expert_ = "MIS"
            self.k_med_ = k_mis
            self.recommended_cluster_number = k_mis
            self.dispatch_margin_signed_ = float(D_mis - D_dcs)
            self.confidence_margin_raw_ = float(abs(H_dcs - H_mis))

            conf = self._dispatch_confidence(
                q_sel=q_mis,
                k_list=k_list,
                k_hat=k_mis,
                D_sel=D_mis,
                D_rej=D_dcs,
            )
            self.confidence_ = conf

            print(f"  - MIS is more decisive ({H_mis:.4f} <= {H_dcs:.4f}). Trusting MIS.")
            print(f"  - Confidence margin: {self.confidence_margin_raw_:.4f}")
            print(f"  - MED Confidence: {self.confidence_:.4f}")
            print(f"\nFinal Optimal k from MED: {k_mis}")
            return int(k_mis)

