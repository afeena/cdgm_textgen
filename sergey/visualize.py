import torch
from data_loading_utils import Embedding


def main():
    embedding_baseline = Embedding(
        base_path='/ivi/ilps/projects/cnmt/knn-mt-dispersion/datastores/model_512_no_dispersion/',
        features_path='train.features.npy',
        yaml_path="410.roen.faiss_L2_IVFPQindex_cs64_nprobe32_d512_nc4096_metadata.yaml",
        faiss_path='410.roen.faiss_L2_IVFPQindex_cs64_nprobe32_d512_nc4096'
    )
    embedding_dispersion = Embedding(
        base_path='/ivi/ilps/projects/cnmt/knn-mt-dispersion/datastores/model_512_dispersion/',
        features_path='582.train.features.npy',
        yaml_path="582.roen.faiss_L2_IVFPQindex_cs64_nprobe32_d512_nc2048_meta.yaml",
        faiss_path='582.roen.faiss_L2_IVFPQindex_cs64_nprobe32_512_4096'
    )
    embedding_baseline.load_raw()
    embedding_dispersion.load_raw()

    if torch.cuda.is_available():
        gpu=True
    else:
        gpu=False

    gpu = False
    print(f"Loading FAISS index on {'GPU' if gpu else 'CPU'}")
    embedding_baseline.load_faiss(gpu=gpu)
    embedding_dispersion.load_faiss(gpu=gpu)

    print(embedding_dispersion.knn_index.nlist)
    print(embedding_baseline.knn_index.nlist)

    Ds_baseline = embedding_baseline.average_k_neares_neighbour_distance(n=1000, k=10, use_faiss_distances=True, speedtest=True)
    Ds_dispersion = embedding_dispersion.average_k_neares_neighbour_distance(n=1000, k=10, use_faiss_distances=True, speedtest=True)

    print(Ds_baseline)
    print(Ds_dispersion)


if __name__ == "__main__":
    main()
