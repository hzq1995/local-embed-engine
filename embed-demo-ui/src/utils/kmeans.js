/**
 * Pure-JS K-means clustering with k-means++ initialization.
 * Works on n-dimensional float arrays. No external dependencies.
 *
 * @param {number[][]} vectors   - array of n-dim vectors
 * @param {number} k             - number of clusters
 * @param {number} maxIter       - max iterations (default 100)
 * @returns {{ labels: number[], centroids: number[][] }}
 */
export function kMeans(vectors, k, maxIter = 100) {
  const n = vectors.length;
  const dim = vectors[0].length;

  if (n === 0 || k <= 0) return { labels: [], centroids: [] };
  if (k >= n) {
    // each point is its own cluster
    return {
      labels: vectors.map((_, i) => i % k),
      centroids: vectors.slice(0, k).map((v) => [...v]),
    };
  }

  // ── k-means++ initialization ────────────────────────────────────
  const centroids = [];
  // pick first centroid uniformly at random
  centroids.push([...vectors[Math.floor(Math.random() * n)]]);

  for (let c = 1; c < k; c++) {
    // compute squared distances from each point to nearest centroid
    const dists = vectors.map((v) => {
      let minD2 = Infinity;
      for (const cen of centroids) {
        const d2 = sqDist(v, cen, dim);
        if (d2 < minD2) minD2 = d2;
      }
      return minD2;
    });

    const total = dists.reduce((a, b) => a + b, 0);
    // weighted random selection
    let rand = Math.random() * total;
    let chosen = n - 1;
    for (let i = 0; i < n; i++) {
      rand -= dists[i];
      if (rand <= 0) {
        chosen = i;
        break;
      }
    }
    centroids.push([...vectors[chosen]]);
  }

  // ── main loop ────────────────────────────────────────────────────
  let labels = new Array(n).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    // assignment step
    let changed = false;
    const newLabels = vectors.map((v, i) => {
      let bestC = 0;
      let bestD = sqDist(v, centroids[0], dim);
      for (let c = 1; c < k; c++) {
        const d = sqDist(v, centroids[c], dim);
        if (d < bestD) {
          bestD = d;
          bestC = c;
        }
      }
      if (bestC !== labels[i]) changed = true;
      return bestC;
    });
    labels = newLabels;

    if (!changed) break;

    // update step
    const sums = Array.from({ length: k }, () => new Float64Array(dim));
    const counts = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      counts[c]++;
      for (let d = 0; d < dim; d++) sums[c][d] += vectors[i][d];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) {
        // reinitialize empty centroid to a random point
        centroids[c] = [...vectors[Math.floor(Math.random() * n)]];
      } else {
        for (let d = 0; d < dim; d++) {
          centroids[c][d] = sums[c][d] / counts[c];
        }
      }
    }
  }

  return { labels, centroids: centroids.map((c) => Array.from(c)) };
}

function sqDist(a, b, dim) {
  let s = 0;
  for (let d = 0; d < dim; d++) {
    const diff = a[d] - b[d];
    s += diff * diff;
  }
  return s;
}

/**
 * Assign each vector to the nearest centroid.
 * Used to label all grid points after k-means trains on a subset.
 *
 * @param {number[][]} vectors   - all grid-point embeddings
 * @param {number[][]} centroids - cluster centroids from kMeans()
 * @returns {number[]} cluster label per vector
 */
export function assignToCentroids(vectors, centroids) {
  const k = centroids.length;
  const dim = centroids[0]?.length ?? 0;
  return vectors.map((v) => {
    let bestC = 0;
    let bestD = Infinity;
    for (let c = 0; c < k; c++) {
      const d = sqDist(v, centroids[c], dim);
      if (d < bestD) {
        bestD = d;
        bestC = c;
      }
    }
    return bestC;
  });
}
