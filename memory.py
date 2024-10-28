import os
import random
from sklearn.metrics import silhouette_score

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from eval_utils import *
import diversipy


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
        """
        Reservoir sampling algorithm.
        :param num_seen_examples: the number of seen examples
        :param buffer_size: the maximum buffer size
        :return: the target index if the current image is sampled, else -1
        """
        if num_seen_examples < buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < buffer_size:
            return rand
        else:
            return -1


class Memory(object):
    def __init__(self, opt):
        """
        Init memory.
        Args:
            opt.mem_max_classes: int, max number of (pseudo-)classes in memory
            opt.size_per_class: int, number of samples per class
            opt.mem_update_type: str, memory update type, choices are:
                'rdn' (random)
                'mo_rdn' (momentum random),
                'reservoir' (reservoir sampling)
                'simil' (similarity-based selection)
            opt.mem_update_class_based: bool, whether perform clustering and update
                for each cluster/class
            opt.mem_max_new_ratio: float, max ratio of new samples
                if 'mo_rdn' is selected for mem update type
        """
        self.max_classes = opt.mem_max_classes
        self.max_size = opt.mem_size
        self.size_per_class = self.max_size // self.max_classes
        self.sample_type = opt.mem_update_type
        self.cluster_type = opt.mem_cluster_type
        self.max_new_ratio = opt.mem_max_new_ratio
        self.images = []  # A list of numpy arrays
        self.labels_set = []  # Pseud labels assisting memory update
        self.true_labels = []  # Same organization as self.images for true labels record
        self.save_folder = opt.save_folder
        self.update_cnt = 0
        self.num_seen_examples = 0

        # set seed for reproducing
        random.seed(opt.trial)

    def sampling(self, lb, old_sz, new_sz, sz_per_lb):
        """
        Implementation of various sampling methods.

        Args:
            lb: int, ground-truth label or pseudo label of the class
            old_sz: int, size of old data samples
            new_sz: int, size of new data samples
            sz_per_lb: int, upperbound on size of samples per label/class,
                take self.size_per_class with class-based sampling,
                take self.max_size without class-based sampling

        Return:
            select_ind: numpy array of the list of indices that are selected in
                the ind th memory bin
        """
        ind = self.labels_set.index(lb)
        select_ind = np.arange(old_sz + new_sz)
        # Memory Update - sample selection
        if old_sz + new_sz > sz_per_lb:
            if self.sample_type == 'rdn':
                select_ind = np.random.choice(old_sz + new_sz, sz_per_lb,
                                              replace=False)
                self.images[ind] = self.images[ind][select_ind]
                self.true_labels[ind] = self.true_labels[ind][select_ind]
            elif self.sample_type == 'mo_rdn':
                num_new_samples = min(new_sz, int(sz_per_lb * self.max_new_ratio))
                num_old_samples = max(int(sz_per_lb * (1 - self.max_new_ratio)),
                    sz_per_lb - num_new_samples)
                num_old_samples = min(old_sz, num_old_samples)
                select_ind_old = np.random.choice(old_sz, num_old_samples,
                                                  replace=False)
                select_ind_new = old_sz + np.random.choice(new_sz, num_new_samples,
                                                           replace=False)
                select_ind = np.concatenate((select_ind_old, select_ind_new), axis=0)
                self.images[ind] = self.images[ind][select_ind]
                self.true_labels[ind] = self.true_labels[ind][select_ind]
            elif self.sample_type == 'reservoir':
                select_ind = list(np.arange(sz_per_lb))
                cur_ind = np.arange(sz_per_lb)  # Use to record the original index
                for i in range(sz_per_lb, old_sz + new_sz):
                    # i corresponds to the extra portion
                    index = reservoir(self.num_seen_examples, sz_per_lb)
                    if index >= 0:
                        self.images[ind][index] = self.images[ind][i]
                        self.true_labels[ind][index] = self.true_labels[ind][i]
                        select_ind.remove(cur_ind[index])
                        cur_ind[index] = i
                        select_ind.append(i)

                self.images[ind] = self.images[ind][:sz_per_lb]
                self.true_labels[ind] = self.true_labels[ind][:sz_per_lb]
                select_ind = np.array(select_ind)
            elif self.sample_type == 'simil':
                num_new_samples = min(new_sz, int(sz_per_lb * self.max_new_ratio))
                num_old_samples = max(int(sz_per_lb * (1 - self.max_new_ratio)),
                                      sz_per_lb - num_new_samples)
                num_old_samples = min(old_sz, num_old_samples)

                simil_sum = np.sum(self.similarity_matrix[ind], axis=1)
                select_ind_old = (-simil_sum[:old_sz]).argsort()[:num_old_samples]
                select_ind_new = old_sz + (-simil_sum[old_sz:]).argsort()[:num_new_samples]

                select_ind = np.concatenate((select_ind_old, select_ind_new),
                                            axis=0)
                self.images[ind] = self.images[ind][select_ind]
                self.true_labels[ind] = self.true_labels[ind][select_ind]
            else:
                raise ValueError(
                    'memory update policy not supported: {}'.format(self.sample_type))

        return select_ind

    def update_w_labels(self, new_images, new_labels):
        """
        Update memory samples.
        No need to check the number of classes if labels are provided.
        Args:
            new_images: torch array, new incoming images
            new_labels: torch array, new ground-truth labels
        """
        new_images = new_images.detach().numpy()
        new_labels = new_labels.detach().numpy()
        new_labels_set = set(new_labels)
        self.num_seen_examples += new_images.shape[0]

        for lb in new_labels_set:
            new_ind = (np.array(new_labels) == lb)
            new_sz = np.sum(new_ind)
            if lb in self.labels_set:  # already seen
                ind = self.labels_set.index(lb)
                old_sz = self.images[ind].shape[0]
                self.images[ind] = np.concatenate(
                    (self.images[ind], new_images[new_ind]),
                    axis=0)
                self.true_labels[ind] = np.concatenate(
                    (self.true_labels[ind], new_labels[new_ind]),
                    axis=0)
            else:  # first-time seen labels
                self.labels_set.append(lb)
                old_sz = 0
                self.images.append(new_images[new_ind])
                self.true_labels.append(new_labels[new_ind])

            # Memory update - sample selection
            # The key is transfer lb - the ground-truth label,
            # and sz_per_lb - size upperbound for each class
            self.sampling(lb, old_sz, new_sz, self.size_per_class)

    def update_wo_labels(self, new_images, new_labels, model=None):
        """
        Update memory samples.
        Args:
            new_images: torch array, new incoming images
            new_labels: torch array, new ground-truth labels, only keep for record
            model: network model being trained, used in kmeans and spectral cluster type

        Return:
            select_indices: numpy array of selected indices in all_images
        """
        new_images = new_images.detach().numpy()
        new_labels = new_labels.detach().numpy()
        self.num_seen_examples += new_images.shape[0]

        if len(self.images) > 0:  # Not first-time insertion
            old_images, old_labels = self.get_mem_samples_w_true_labels()
            old_images = old_images.detach().numpy()
            old_labels = old_labels.detach().numpy()
            old_sz = old_images.shape[0]
            all_images = np.concatenate((old_images, new_images), axis=0)
            all_true_labels = np.concatenate((old_labels, new_labels), axis=0)
        else:  # first-time insertion
            old_sz = 0
            all_images = new_images
            all_true_labels = new_labels

        # Create a binary indicator of whether the image is an old or new sample
        old_ind = np.zeros(all_images.shape[0], dtype=np.bool)
        old_ind[:old_sz] = 1

        # Get latent embeddings
        feed_images = torch.from_numpy(all_images)
        if torch.cuda.is_available():
            feed_images = feed_images.cuda(non_blocking=True)
        all_embeddings = model(feed_images).detach().cpu().numpy()
        # all_embeddings_mean = np.mean(all_embeddings, axis=0, keepdims=True)
        # all_embeddings = (all_embeddings - all_embeddings_mean) * 1e4

        if self.cluster_type == 'none':
            # One big cluster for all samples
            # The key is transfer lb - the pseudo label
            # and sz_per_lb - size upperbound
            new_sz = all_images.shape[0] - old_sz
            self.images = [all_images]
            self.true_labels = [all_true_labels]
            self.labels_set = [0]

            # Memory update - sample selection
            select_indices = self.sampling(0, old_sz, new_sz, self.max_size)

        elif self.cluster_type in ['kmeans', 'spectral']:
            # Clustering
            simil_matrix = tsne_simil(all_embeddings, metric='cosine')

            if self.cluster_type == 'kmeans':
                if old_sz > 0:
                    cluster = KMeans(n_clusters=self.max_classes).fit(all_embeddings[old_ind])
                    pred_lb = cluster.predict(all_embeddings)
                else:
                    pred_lb = KMeans(n_clusters=self.max_classes).fit_predict(all_embeddings)
            elif self.cluster_type == 'spectral':
                pred_lb = SpectralClustering(n_clusters=self.max_classes, affinity='precomputed',
                                             n_init=10, assign_labels='discretize').fit_predict(simil_matrix)

            self.images, self.true_labels, self.similarity_matrix = [], [], []
            self.labels_set = list(np.unique(pred_lb))

            # Record the indices of selected images from all_images
            select_indices = []

            # Calculate tsne similarity
            for lb in self.labels_set:
                lb_ind = (pred_lb == lb)
                self.images.append(all_images[lb_ind])
                self.true_labels.append(all_true_labels[lb_ind])
                self.similarity_matrix.append(simil_matrix[lb_ind][:, lb_ind])
                old_sz = np.sum(np.logical_and(old_ind, lb_ind))
                new_sz = np.sum(lb_ind) - old_sz

                # Memory update - sample selection
                print(lb, old_sz, new_sz, self.size_per_class)
                select_ind = self.sampling(lb, old_sz, new_sz, self.size_per_class)
                select_indices += list(np.arange(all_images.shape[0])[lb_ind][select_ind])

            select_indices = np.array(sorted(select_indices))

        elif self.cluster_type in ['max_coverage', 'psa', 'maximin', 'energy']:
            # Clustering
            simil_matrix = tsne_simil(all_embeddings, metric='cosine')

            # Init selected indices as all indices
            select_indices = np.arange(all_embeddings.shape[0])

            if all_embeddings.shape[0] > self.max_size:  # needs subset selection
                if self.cluster_type == 'max_coverage':
                    # simil_min = np.min(simil_matrix)
                    simil_max = np.max(simil_matrix)
                    simil_mean = np.mean(simil_matrix)
                    simil_threshold = simil_mean + 0.2 * (simil_max - simil_mean)
                    simil_mask = np.zeros_like(simil_matrix)
                    simil_mask[simil_matrix > simil_threshold] = 1
                    print('avg masks: {} all images: {}'.format(
                        np.sum(simil_mask) / simil_mask.shape[0], simil_mask.shape[0]
                    ))

                    sorted_idx = np.argsort(simil_mask.sum(axis=1))
                    select_indices = sorted_idx[:self.max_size]

                elif self.cluster_type == 'psa':
                    select_indices = diversipy.subset.psa_select(all_embeddings, self.max_size)
                    select_indices.sort()
                    print('select {} from {}'.format(self.max_size, all_embeddings.shape[0]))

                elif self.cluster_type == 'maximin':
                    # As in the MinRed paper, use cosine similarity after normed
                    select_indices = diversipy.subset.select_greedy_maximin(
                        all_embeddings, self.max_size, dist_args={'dist': 'cosine'}
                    )
                    select_indices.sort()
                    print('select {} from {}'.format(self.max_size, all_embeddings.shape[0]))

                else:  # self.cluster_type == 'energy':
                    select_indices = diversipy.subset.select_greedy_energy(all_embeddings, self.max_size)
                    select_indices.sort()
                    print('select {} from {}'.format(self.max_size, all_embeddings.shape[0]))

            self.images = [all_images[select_indices]]
            self.true_labels = [all_true_labels[select_indices]]
            self.labels_set = [0]

        else:
            raise ValueError(
                'memory cluster policy not supported: {}'.format(self.cluster_type))

        return all_embeddings, all_true_labels, select_indices

    def fullfill(self):
        """Fill in the complete memory by duplication"""
        if len(self.labels_set) <= 0:  # empty memory
            return
        for lb in self.labels_set:
            ind = self.labels_set.index(lb)
            cur_len = self.images[ind].shape[0]

            if cur_len < self.size_per_class:  # repeat until overflow
                mul = np.ceil(self.size_per_class * 1. / cur_len)
                self.images[ind] = self.images[ind].repeat(mul, axis=0)

            if self.images[ind].shape[0] > self.size_per_class:  # cut
                self.images[ind] = self.images[ind][:self.size_per_class]

    def get_mem_samples(self):
        """
        Combine all stored samples and pseudo labels.
        Returns:
            images: numpy array of all images, (sample #, image)
            labels: numpy array of all pseudo labels, (sample #, pseudo label)
        If updated with update_w_labels, the returned labels are the ground-truth labels.
        If updated with update_wo_labels, the returned labels are the pseudo labels.
        """
        images, labels = None, None
        for lb in self.labels_set:
            ind = self.labels_set.index(lb)
            if images is None:  # First label
                images = self.images[ind]
                labels = np.repeat(lb, self.images[0].shape[0])
            else:  # Subsequent labels to be concatenated
                images = np.concatenate((images, self.images[ind]), axis=0)
                labels = np.concatenate((labels,
                                         np.repeat(lb, self.images[ind].shape[0])),
                                        axis=0)

        if images is None:  # Empty memory
            return None, None
        else:
            return torch.from_numpy(images), torch.from_numpy(labels)

    def get_mem_samples_w_true_labels(self):
        """
        Combine all stored samples and true labels.
        Returns:
            images: numpy array of all images, (sample #, image)
            labels: numpy array of all true labels, (sample #, true label)
        """
        images, labels = None, None
        for lb in self.labels_set:
            ind = self.labels_set.index(lb)
            if images is None:  # First label
                images = self.images[ind]
                labels = self.true_labels[ind]
            else:  # Subsequent labels to be concatenated
                images = np.concatenate((images, self.images[ind]), axis=0)
                labels = np.concatenate((labels, self.true_labels[ind]), axis=0)

        if images is None:  # Empty memory
            return None, None
        else:
            return torch.from_numpy(images), torch.from_numpy(np.array(labels))

    def k_cluster_sil(self, x_mem, x_all, y_all, candidate_k, similarity_matrix):
        """
        Use silhouette score to find the optimal k for clustering
        Args:
            x_mem: numpy array, memory data points to be clustered
            x_all: numpy array, mem and streaming data points to be predicted
            y_all: numpy array, true labels of x_all
            candidate_k: list, candidate number of clusters
            similarity_matrix: numpy matrix, pairwise tsne similarity matrix
        Return:
            opt_k: int, optimal number of clusters, or k
            opt_labels: list, predicted labels of x_all under the optimal number of classes
        """
        sil = []
        pred_labels = []
        similarity_mem = similarity_matrix[:x_mem.shape[0], :x_mem.shape[0]]
        for k in candidate_k:
            k_cluster = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, verbose=0)
            # labels = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='average').fit_predict(x)
            # k_cluster = SpectralClustering(n_clusters=k, affinity='precomputed',
            #                               n_init=10, assign_labels='discretize')
            if x_mem.shape[0] > 0:
                labels_mem = k_cluster.fit_predict(x_mem)
                sil.append(silhouette_score(x_mem, labels_mem, metric='cosine'))
                pred_labels.append(k_cluster.predict(x_all))
            else:  # Deal with the first run with empty memory
                labels_all = k_cluster.fit_predict(x_all)
                sil.append(silhouette_score(x_all, labels_all, metric='cosine'))
                pred_labels.append(labels_all)

        print(candidate_k)
        print(sil)
        print(self.sil_offset)
        sil[0] += self.sil_offset
        opt_ind = np.array(sil).argsort()[-1]  # Last index is for the max element
        opt_k, opt_labels = candidate_k[opt_ind], pred_labels[opt_ind]
        print('Opt k during mem update: {}'.format(opt_k))

        if self.update_cnt % 5 == 0:
            plot_tsne(x_all, opt_labels, y_all,
                      fig_name=os.path.join(self.save_folder, 'sil_mem_{}.png'.format(self.update_cnt)))
        self.update_cnt += 1

        return opt_k, opt_labels