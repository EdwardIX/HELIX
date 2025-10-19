import datasets
import torch
import torch.nn.functional as F
import numpy as np
import json
import os

from recipe.algo_evolve.dataset.embedding_modules import PythonEmbeddingModule, YamlEmbeddingModule

def get_score_from_dataframe(dataframe:datasets.Dataset):
    scores = []
    for item in dataframe:
        if 'reward' in item:
            scores.append(item['reward'])
        elif 'score' in item:
            scores.append(item['score'])
        elif 'extra_info' in item and 'score' in item['extra_info']:
            scores.append(item['extra_info']['score'])
        elif 'extra_info' in item and 'reward' in item['extra_info']:
            scores.append(item['extra_info']['reward'])
        else:
            raise ValueError("Data item must contain either 'reward' or 'score'.")

    return np.array(scores, dtype=np.float32)

def get_solution_from_dataframe(dataframe:datasets.Dataset):
    solutions = []
    for item in dataframe:
        if 'solution' in item:
            solutions.append(item['solution'])
        elif 'extra_info' in item and 'solution' in item['extra_info']:
            solutions.append(item['extra_info']['solution'])
        else:
            raise ValueError("Data item must contain 'solution'.")
    return solutions

class SampleStrategy():
    def __init__(self, dataframe, batchsize):
        self.batchsize = batchsize
        self.mapping = None
    
    def __call__(self, idx):
        assert idx < self.batchsize, "Index out of bounds for the sample strategy."
        return int(self.mapping[idx % len(self.mapping)]) # Avoid out of bounds
    
    def extend(self, new_dataframe):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def log(self, log_path):
        pass
    
class RandomSampleStrategy(SampleStrategy):
    def __init__(self, dataframe, batchsize):
        super().__init__(dataframe, batchsize)
        self.current_length = len(dataframe)
        self._generate_mapping()
    
    def _generate_mapping(self):
        self.mapping = np.random.choice(self.current_length, size=min(self.current_length, self.batchsize), replace=False)

    def extend(self, new_dataframe):
        self.current_length += len(new_dataframe)
        self._generate_mapping()
        return new_dataframe

class TopKSampleStrategy(SampleStrategy):
    def __init__(self, dataframe, batchsize):
        super().__init__(dataframe, batchsize)
        self.scores = get_score_from_dataframe(dataframe)
        self._generate_mapping()
    
    def _generate_mapping(self):
        if len(self.scores) < self.batchsize:
            self.mapping = np.arange(len(self.scores))
        else:
            self.mapping = np.argpartition(self.scores, -self.batchsize)[-self.batchsize:]

    def extend(self, new_dataframe):
        self.scores = np.concatenate([self.scores, get_score_from_dataframe(new_dataframe)])
        self._generate_mapping()
        return new_dataframe

class TopKDivSampleStrategy(SampleStrategy):
    def __init__(self, dataframe, batchsize, nsga_config={}):
        super().__init__(dataframe, batchsize)
        self.config = nsga_config

        self.embedding_module = {
            'PythonEmbeddingModule': PythonEmbeddingModule,
            'YamlEmbeddingModule': YamlEmbeddingModule
        }[nsga_config.get('embedding_module', 'PythonEmbeddingModule')](
            model_name=nsga_config.get('embed_model', 'BAAI/bge-code-v1'),
            batch_size=nsga_config.get('embed_batch_size', 1024),
            max_len=nsga_config.get('embed_max_len', 1024)
        )

        self.knn_avg = nsga_config.get('knn_avg', 5)
        self.scores = get_score_from_dataframe(dataframe)
        self.codes = get_solution_from_dataframe(dataframe)
        self.emb_cache = None

        self.mapping, self.extra_info = self.compute_ranking(self.codes, batchsize)

    def extend(self, new_dataframe):
        codes_new = get_solution_from_dataframe(new_dataframe)
    
        self.scores = np.concatenate([self.scores, get_score_from_dataframe(new_dataframe)])
        self.codes = self.codes + codes_new
        self.mapping, self.extra_info = self.compute_ranking(self.codes, self.batchsize)
        return new_dataframe

    def log(self, log_path):
        """Log code, scores and diversity score in one json file"""
        with open(os.path.join(log_path, f'nsga_{len(self.codes)}.json'), 'w') as f:
            json.dump([
                {"code": c, "score": float(s), "diversity_score": float(ds)}
                for c, s, ds in zip(self.codes, self.scores, self.extra_info['diversity_score'])
            ], f)
    
    @torch.no_grad()
    def compute_diversity_score(self, pop: list[str], device=torch.device('cuda')):
        if self.emb_cache is None:
            self.emb_cache = torch.zeros(0, self.embedding_module.embed_dim) # empty placeholder

        self.embedding_module.load()

        if len(pop) > len(self.emb_cache):
            codes_to_embed = [pop[i] for i in range(len(self.emb_cache), len(pop))]
            new_vecs = self.embedding_module.encode(codes_to_embed, device=device)
            self.emb_cache = torch.cat([self.emb_cache] + [new_vecs.cpu()], dim=0)

        M = self.emb_cache.to(device)          # (N, H) on GPU
        sim = torch.mm(M, M.t())                    # (N, N) cosine similarity
        dist = 1.0 - sim
        dist.fill_diagonal_(float("inf"))
        knn_dist, _ = torch.topk(dist, k=min(self.knn_avg, dist.shape[0]), dim=1, largest=False)
        mean_knn_dist = knn_dist.mean(dim=1).cpu().tolist()
        del M

        self.embedding_module.release()

        return mean_knn_dist
    
    def compute_ranking(self, pool: list[str], next_gen_size):
        div_scores = self.compute_diversity_score(pool)
        ranked_indices = np.argsort(div_scores)[-next_gen_size:]
        return ranked_indices, {"diversity_score": div_scores}

class NSGASampleStrategy(SampleStrategy):
    def __init__(self, dataframe, batchsize, nsga_config={}):
        super().__init__(dataframe, batchsize)
        self.config = nsga_config

        self.embedding_module = {
            'PythonEmbeddingModule': PythonEmbeddingModule,
            'YamlEmbeddingModule': YamlEmbeddingModule
        }[nsga_config.get('embedding_module', 'PythonEmbeddingModule')](
            model_name=nsga_config.get('embed_model', 'BAAI/bge-code-v1'),
            batch_size=nsga_config.get('embed_batch_size', 1024),
            max_len=nsga_config.get('embed_max_len', 1024)
        )

        self.knn_avg = nsga_config.get('knn_avg', 5)
        self.scores = get_score_from_dataframe(dataframe)
        self.codes = get_solution_from_dataframe(dataframe)
        self.emb_cache = None

        self.mapping, self.extra_info = self.compute_nsga_ranking(self.codes, batchsize)

    def extend(self, new_dataframe):
        codes_new = get_solution_from_dataframe(new_dataframe)
    
        self.scores = np.concatenate([self.scores, get_score_from_dataframe(new_dataframe)])
        self.codes = self.codes + codes_new
        self.mapping, self.extra_info = self.compute_nsga_ranking(self.codes, self.batchsize)
        return new_dataframe

    def log(self, log_path):
        """Log code, scores and diversity score in one json file"""
        with open(os.path.join(log_path, f'nsga_{len(self.codes)}.json'), 'w') as f:
            json.dump([
                {"code": c, "score": float(s), "diversity_score": float(ds)}
                for c, s, ds in zip(self.codes, self.scores, self.extra_info['diversity_score'])
            ], f)
        with open(os.path.join(log_path, f'nsga_emb_cache_{len(self.codes)}.pt'), 'wb') as f:
            torch.save(self.emb_cache, f)

    def compute_nsga_ranking(self, pool: list[str], next_gen_size):
        f1p = self.scores
        f2p = self.compute_diversity_score(pool)
        fronts_p = self._fast_sort(f1p, f2p)
        crowd_p  = [self._crowding(f1p, f2p, fr) for fr in fronts_p]
        
        next_idx = []
        for fi, fr in enumerate(fronts_p):
            if len(next_idx) + len(fr) < next_gen_size:
                next_idx.extend(fr)
                # print(fr, next_idx)
            else:
                order = sorted(range(len(fr)), key=lambda j: crowd_p[fi][j], reverse=True)
                for j in order:
                    next_idx.append(fr[j])
                    # print(fr[j], next_idx)

                    if len(next_idx) == next_gen_size:
                        break
                break
       
        # pop = [pool[i] for i in next_idx]
        
        return next_idx, {"diversity_score": f2p}
    
    @torch.no_grad()
    def compute_diversity_score(self, pop: list[str], device=torch.device('cuda')):
        if self.emb_cache is None:
            self.emb_cache = torch.zeros(0, self.embedding_module.embed_dim) # empty placeholder

        self.embedding_module.load()

        if len(pop) > len(self.emb_cache):
            codes_to_embed = [pop[i] for i in range(len(self.emb_cache), len(pop))]
            new_vecs = self.embedding_module.encode(codes_to_embed, device=device)
            self.emb_cache = torch.cat([self.emb_cache] + [new_vecs.cpu()], dim=0)

        M = self.emb_cache.to(device)          # (N, H) on GPU
        sim = torch.mm(M, M.t())                    # (N, N) cosine similarity
        dist = 1.0 - sim
        dist.fill_diagonal_(float("inf"))
        knn_dist, _ = torch.topk(dist, k=min(self.knn_avg, dist.shape[0]), dim=1, largest=False)
        mean_knn_dist = knn_dist.mean(dim=1).cpu().tolist()
        del M

        self.embedding_module.release()

        return mean_knn_dist

    def _fast_sort(self, values1, values2):
        S=[[] for i in range(0,len(values1))]
        front = [[]]
        n=[0 for i in range(0,len(values1))]
        rank = [0 for i in range(0, len(values1))]

        for p in range(0,len(values1)):
            S[p]=[]
            n[p]=0
            for q in range(0, len(values1)):
                if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                    if q not in S[p]:
                        S[p].append(q)
                elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                    n[p] = n[p] + 1
            if n[p]==0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)

        i = 0
        while(front[i] != []):
            Q=[]
            for p in front[i]:
                for q in S[p]:
                    n[q] =n[q] - 1
                    if( n[q]==0):
                        rank[q]=i+1
                        if q not in Q:
                            Q.append(q)
            i = i+1
            front.append(Q)

        del front[len(front)-1]
        return front

    def _crowding(self, v1, v2, fr):
        l=len(fr); D=[0.0]*l
        if l==0: return D
        order1=sorted(range(l), key=lambda i:v1[fr[i]])
        order2=sorted(range(l), key=lambda i:v2[fr[i]])
        D[order1[0]]=D[order1[-1]]=D[order2[0]]=D[order2[-1]]=float('inf')
        max_v1, max_v2, min_v1, min_v2 = max(v1), max(v2), min(v1), min(v2)
        if max_v1!=min_v1:
            for k in range(1,l-1):
                D[order1[k]]+=(v1[fr[order1[k+1]]]-v1[fr[order1[k-1]]])/(max_v1-min_v1)
        if max_v2!=min_v2:
            for k in range(1,l-1):
                D[order2[k]]+=(v2[fr[order2[k+1]]]-v2[fr[order2[k-1]]])/(max_v2-min_v2)
        return D
