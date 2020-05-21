from scipy.io import loadmat
import numpy as np
ppi_network_dict = loadmat('ppi_network.mat')  # (i, j)表示第i个基因和第j个基因是否有关系
g_p_network_dict = loadmat('g_p_network.mat')  # (i, j)表示第i个基因和第j个疾病是否有关系
# 疾病表型的相似关系 phenotype_name 对应ID，phenotype_network对应相似度
phenotype_network_dict = loadmat('phenotype_network.mat')
# print(ppi_network_dict)  # (ppi_network, gene_name)
# print(g_p_network_dict)  # (g_p_network)
# print(phenotype_network_dict)  # (phenotype_network, phenotype_name)
ppi_network = ppi_network_dict['ppi_network']
g_p_network = g_p_network_dict['g_p_network']
phenotype_network_id = phenotype_network_dict['phenotype_network'][:, 0].tolist()
phenotype_network = phenotype_network_dict['phenotype_network'][:, 1:]
print(ppi_network.shape, g_p_network.shape,
      phenotype_network.shape, len(phenotype_network_id))
# Normalize PPI network column
ppi_network_norm = ppi_network / np.sum(ppi_network, axis=1).reshape((
    ppi_network.shape[0], 1
))
print(ppi_network_norm)
# initialize the query phenotype with genes
input_id = int(input('Please input id: '))
while input_id not in phenotype_network_id:
    input_id = int(input('Please input the right id: '))
# enrich query phenotype with its 5 most similar phenotype
idx = phenotype_network_id.index(input_id)
print(idx)
phenotype_for_id = phenotype_network[idx].tolist()
highest_similarity = sorted(phenotype_for_id, reverse=True)[:5]
print(highest_similarity)
# similar_phenotypes = [phenotype_for_id.index(h) for h in highest_similarity]
similar_phenotypes = []
for h in highest_similarity:
    pid = phenotype_for_id.index(h)
    while pid in similar_phenotypes:
        pid = phenotype_for_id[pid + 1:].index(h) + pid + 1
    similar_phenotypes.append(pid)
print(similar_phenotypes)
# represent query phenotype with its casual genes
g_p_list = g_p_network[:, similar_phenotypes]
initial_gene = np.argwhere(g_p_list != 0)[:, 0].tolist()
print(initial_gene)
# propagate similarity through gene network by RWR
alpha = 0.1
iteration = 50
N = ppi_network_norm.shape[1]
R = np.zeros((N, 1), dtype=np.float)
R[initial_gene, 0] = 1.0
bias = R
for i in range(iteration):
    R = (1 - alpha) * bias + alpha * (ppi_network_norm @ R)
# choose the top 10 most probable results
R = R.flatten().tolist()
result_ranking = sorted(R, reverse=True)[:10]
print(result_ranking)
ranking_genes = []
for ranking in result_ranking:
    Ridx = R.index(ranking)
    while Ridx in ranking_genes:
        Ridx = R[Ridx + 1:].index(ranking) + Ridx + 1
    ranking_genes.append(Ridx)
print(ranking_genes)
gene_name_list = ppi_network_dict['gene_name']
for gene_index, gene_name in zip(ranking_genes, gene_name_list[ranking_genes][:, 0]):
    print(gene_index + 1, gene_name[0])
