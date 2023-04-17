import logging
import os
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.kmeans import RiemannianKMeans

os.environ['GEOMSTATS_BACKEND'] = "pytorch"

logger = logging.getLogger("cdgm_textgen.modules.cluster_embeddings")


def cluster_embeddings_riemannian(embeddings, n_clusters):
    sphere = Hypersphere(dim=embeddings.shape[1])

    logger.info("Clustering embeddings on the hypersphere with {} clusters".format(n_clusters))

    kmeans = RiemannianKMeans(metric=sphere.metric,
                              n_clusters=n_clusters,
                              tol=1e-5,
                              verbose=2,
                              max_iter=1000)
    kmeans.fit(embeddings)
    cluster_centroids = kmeans.centroids
    return cluster_centroids
