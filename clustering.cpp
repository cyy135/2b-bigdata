#include "clustering.h"
#include <iostream>
#include <random>
#include <map>
#include <set>
#include <queue>
#include <unordered_map>

using namespace std;

// Adjusts the number of clusters to targetClusters and assigns noise points
vector<int> adjustClusterNumber(const vector<vector<int>>& data, vector<int> clusterAssignments, int targetClusters) {
    // Count clusters (excluding noise)
    int maxCluster = -1;
    for (int label : clusterAssignments) {
        if (label > maxCluster) maxCluster = label;
    }
    int numClusters = maxCluster + 1;

    // If needed, merge or split clusters to reach targetClusters
    if (numClusters != targetClusters) {
        // Collect points for each cluster (excluding noise)
        vector<vector<vector<int>>> clusters(numClusters);
        for (size_t i = 0; i < clusterAssignments.size(); ++i) {
            int label = clusterAssignments[i];
            if (label >= 0) clusters[label].push_back(data[i]);
        }
        // Compute centroids for each cluster
        vector<vector<int>> centroids;
        for (const auto& cluster : clusters) {
            if (cluster.empty()) continue;
            vector<int> centroid(cluster[0].size(), 0);
            for (const auto& pt : cluster) {
                for (size_t d = 0; d < pt.size(); ++d) centroid[d] += pt[d];
            }
            for (size_t d = 0; d < centroid.size(); ++d) centroid[d] /= cluster.size();
            centroids.push_back(centroid);
        }
        // Merge clusters if too many
        if ((int)centroids.size() > targetClusters) {
            vector<pair<int, int>> centroidLabels = hierarchicalClustering(centroids, targetClusters);
            // Map old cluster labels to new merged labels
            vector<int> clusterMap(numClusters, -1);
            int cidx = 0;
            for (int i = 0; i < numClusters; ++i) {
                if (!clusters[i].empty()) {
                    clusterMap[i] = centroidLabels[cidx++].second;
                }
            }
            for (size_t i = 0; i < clusterAssignments.size(); ++i) {
                int label = clusterAssignments[i];
                if (label >= 0) clusterAssignments[i] = clusterMap[label];
            }
        } else if ((int)centroids.size() < targetClusters) {
            // Split largest clusters
            int needed = targetClusters - centroids.size();
            vector<pair<int, int>> clusterSizes; // (size, label)
            for (int i = 0; i < (int)clusters.size(); ++i) {
                if (!clusters[i].empty()) clusterSizes.push_back({(int)clusters[i].size(), i});
            }
            sort(clusterSizes.rbegin(), clusterSizes.rend());
            int nextLabel = centroids.size();
            for (int k = 0; k < needed && k < (int)clusterSizes.size(); ++k) {
                int labelToSplit = clusterSizes[k].second;
                auto& pts = clusters[labelToSplit];
                if ((int)pts.size() < 2) continue;
                // Split this cluster into 2 using hierarchicalClustering
                vector<pair<int, int>> splitLabels = hierarchicalClustering(pts, 2);
                for (size_t i = 0; i < clusterAssignments.size(); ++i) {
                    if (clusterAssignments[i] == labelToSplit) {
                        if (splitLabels.empty()) continue;
                        int splitIdx = i - (i > 0 ? clusterSizes[k-1].first : 0);
                        if (splitIdx < (int)splitLabels.size()) {
                            clusterAssignments[i] = (splitLabels[splitIdx].second == 0) ? labelToSplit : nextLabel;
                        }
                    }
                }
                ++nextLabel;
            }
            // If still not enough clusters, relabel to ensure exactly targetClusters
            vector<int> usedLabels;
            for (int label : clusterAssignments) {
                if (label >= 0 && find(usedLabels.begin(), usedLabels.end(), label) == usedLabels.end())
                    usedLabels.push_back(label);
            }
            if ((int)usedLabels.size() > targetClusters) {
                // Merge again if overshot
                vector<vector<int>> finalCentroids;
                for (int lbl : usedLabels) {
                    vector<int> centroid(data[0].size(), 0);
                    int count = 0;
                    for (size_t i = 0; i < clusterAssignments.size(); ++i) {
                        if (clusterAssignments[i] == lbl) {
                            for (size_t d = 0; d < data[i].size(); d++) centroid[d] += data[i][d];
                            ++count;
                        }
                    }
                    for (size_t d = 0; d < centroid.size(); ++d) centroid[d] /= count;
                    finalCentroids.push_back(centroid);
                }
                vector<pair<int, int>> finalLabels = hierarchicalClustering(finalCentroids, targetClusters);
                for (size_t i = 0; i < clusterAssignments.size(); ++i) {
                    int idx = find(usedLabels.begin(), usedLabels.end(), clusterAssignments[i]) - usedLabels.begin();
                    if (idx < (int)finalLabels.size())
                        clusterAssignments[i] = finalLabels[idx].second;
                }
            }
        }
    }

    // Assign noise points to nearest cluster centroid
    // 1. Compute centroids of final clusters
    vector<vector<int>> centroids(targetClusters, vector<int>(data[0].size(), 0));
    vector<int> clusterSizes(targetClusters, 0);
    for (size_t i = 0; i < clusterAssignments.size(); ++i) {
        int label = clusterAssignments[i];
        if (label >= 0 && label < targetClusters) {
            for (size_t d = 0; d < data[i].size(); ++d)
                centroids[label][d] += data[i][d];
            clusterSizes[label]++;
        }
    }
    for (int c = 0; c < targetClusters; ++c)
        for (size_t d = 0; d < centroids[c].size(); ++d)
            if (clusterSizes[c] > 0) centroids[c][d] /= clusterSizes[c];

    // 2. Assign noise points to nearest cluster
    for (size_t i = 0; i < clusterAssignments.size(); ++i) {
        if (clusterAssignments[i] == -1) { // noise
            double minDist = 1e20;
            int bestCluster = -1;
            for (int c = 0; c < targetClusters; ++c) {
                double dist = calculateDistance(data[i], centroids[c]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = c;
                }
            }
            clusterAssignments[i] = bestCluster;
        }
    }
    return clusterAssignments;
}

// Function to calculate Euclidean distance between two points
float calculateDistance(const vector<int>& point1, const vector<int>& point2) {
    float sum = 0.0;
    for (size_t i = 0; i < point1.size(); i++) {
        sum += pow(point1[i] - point2[i], 2);
    }
    return sqrt(sum);
}

// Function to find the minimum distance between two clusters
float findMinDistance(const vector<vector<int>>& data, const vector<bool>& cluster1, const vector<bool>& cluster2) {
    float minDist = numeric_limits<float>::max();
    
    for (size_t i = 0; i < data.size(); i++) {
        if (!cluster1[i]) continue;
        for (size_t j = 0; j < data.size(); j++) {
            if (!cluster2[j]) continue;
            float dist = calculateDistance(data[i], data[j]);
            if (dist < minDist) {
                minDist = dist;
            }
        }
    }
    return minDist;
}

// Hierarchical clustering function
vector<pair<int, int>> hierarchicalClustering(const vector<vector<int>>& data, int numClusters) {
    int n = data.size();
    vector<int> clusterAssignments(n);
    
    // Initialize: each point is its own cluster
    for (int i = 0; i < n; i++) {
        clusterAssignments[i] = i;
    }
    
    int currentClusters = n;
    
    // Continue merging until we reach the desired number of clusters
    while (currentClusters > numClusters) {
        float minDist = numeric_limits<float>::max();
        int cluster1 = -1, cluster2 = -1;
        
        // Find the two closest clusters
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (clusterAssignments[i] == clusterAssignments[j]) continue;
                
                // Create temporary cluster membership vectors
                vector<bool> tempCluster1(n, false);
                vector<bool> tempCluster2(n, false);
                
                for (int k = 0; k < n; k++) {
                    if (clusterAssignments[k] == clusterAssignments[i]) {
                        tempCluster1[k] = true;
                    }
                    if (clusterAssignments[k] == clusterAssignments[j]) {
                        tempCluster2[k] = true;
                    }
                }
                
                float dist = findMinDistance(data, tempCluster1, tempCluster2);
                if (dist < minDist) {
                    minDist = dist;
                    cluster1 = clusterAssignments[i];
                    cluster2 = clusterAssignments[j];
                }
            }
        }
        
        // Merge the two closest clusters
        if (cluster1 != -1 && cluster2 != -1) {
            for (int i = 0; i < n; i++) {
                if (clusterAssignments[i] == cluster2) {
                    clusterAssignments[i] = cluster1;
                }
            }
            currentClusters--;
        }
    }
    
    // Renumber clusters to be consecutive starting from 0
    vector<int> uniqueClusters;
    for (int i = 0; i < n; i++) {
        if (find(uniqueClusters.begin(), uniqueClusters.end(), clusterAssignments[i]) == uniqueClusters.end()) {
            uniqueClusters.push_back(clusterAssignments[i]);
        }
    }
    
    vector<int> finalAssignments(n);
    for (int i = 0; i < n; i++) {
        auto it = find(uniqueClusters.begin(), uniqueClusters.end(), clusterAssignments[i]);
        finalAssignments[i] = distance(uniqueClusters.begin(), it);
    }
    
    // Convert to pair format
    vector<pair<int, int>> result;
    for (int i = 0; i < n; i++) {
        result.emplace_back(i, finalAssignments[i]);
    }
    
    return result;
}

// K-means clustering function
vector<pair<int, int>> kmeans(const vector<vector<int>>& data, int numClusters) {
    int n = data.size();
    int dim = data[0].size();
    vector<int> clusterAssignments(n, 0);
    
    // Initialize centroids randomly (using first k points as initial centroids)
    vector<vector<float>> centroids(numClusters, vector<float>(dim));
    for (int i = 0; i < numClusters; i++) {
        for (int j = 0; j < dim; j++) {
            centroids[i][j] = (float)data[i][j];
        }
    }
    
    bool converged = false;
    int maxIterations = 100;
    int iteration = 0;
    
    while (!converged && iteration < maxIterations) {
        converged = true;
        iteration++;
        
        // Assign points to nearest centroid
        for (int i = 0; i < n; i++) {
            float minDist = numeric_limits<float>::max();
            int bestCluster = 0;
            
            for (int k = 0; k < numClusters; k++) {
                float dist = 0.0;
                for (int j = 0; j < dim; j++) {
                    dist += pow(data[i][j] - centroids[k][j], 2);
                }
                dist = sqrt(dist);
                
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = k;
                }
            }
            
            if (clusterAssignments[i] != bestCluster) {
                clusterAssignments[i] = bestCluster;
                converged = false;
            }
        }
        
        // Update centroids
        vector<vector<float>> newCentroids(numClusters, vector<float>(dim, 0.0));
        vector<int> clusterSizes(numClusters, 0);
        
        for (int i = 0; i < n; i++) {
            int cluster = clusterAssignments[i];
            clusterSizes[cluster]++;
            for (int j = 0; j < dim; j++) {
                newCentroids[cluster][j] += data[i][j];
            }
        }
        
        for (int k = 0; k < numClusters; k++) {
            if (clusterSizes[k] > 0) {
                for (int j = 0; j < dim; j++) {
                    centroids[k][j] = newCentroids[k][j] / clusterSizes[k];
                }
            }
        }
    }
    
    cout << "K-means converged after " << iteration << " iterations" << endl;
    
    // Convert to pair format
    vector<pair<int, int>> result;
    for (int i = 0; i < n; i++) {
        result.emplace_back(i, clusterAssignments[i]);
    }
    
    return result;
}

// The main function implementing the original DBSCAN algorithm.
// It uses density-based clustering with eps (neighborhood radius) and minPts (minimum points for core).
// targetClusters is used to adjust the final number of clusters if needed.
std::vector<std::pair<int, int>> dbscan(const std::vector<std::vector<int>>& data, float eps, int minPts, int targetClusters) {
    int n = data.size(); // Number of data points
    if (n == 0 || data[0].empty()) {
        return {}; // Handle empty input data
    }

    std::cout << "[DBSCAN] Starting clustering with eps=" << eps << ", minPts=" << minPts << std::endl;

    // Step 1: Find neighbors for each point
    std::vector<std::vector<int>> neighbors(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                float dist = calculateDistance(data[i], data[j]);
                if (dist <= eps) {
                    neighbors[i].push_back(j);
                }
            }
        }
    }

    // Step 2: Identify core points, border points, and noise points
    std::vector<bool> isCore(n, false);
    std::vector<bool> isNoise(n, false);
    std::vector<int> clusterAssignments(n, -1); // -1 means unassigned

    for (int i = 0; i < n; ++i) {
        if (neighbors[i].size() >= minPts) {
            isCore[i] = true;
        } else {
            isNoise[i] = true; // Initially mark as noise, will be updated if it becomes border
        }
    }

    // Step 3: Perform DBSCAN clustering
    int currentCluster = 0;
    std::vector<bool> visited(n, false);

    for (int i = 0; i < n; ++i) {
        if (visited[i] || isNoise[i]) continue;

        if (isCore[i]) {
            // Start a new cluster
            std::queue<int> q;
            q.push(i);
            visited[i] = true;
            clusterAssignments[i] = currentCluster;

            // Expand the cluster using BFS
            while (!q.empty()) {
                int current = q.front();
                q.pop();

                // Add all neighbors to the cluster
                for (int neighbor : neighbors[current]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        clusterAssignments[neighbor] = currentCluster;
                        
                        // If neighbor is a core point, add its neighbors to the queue
                        if (isCore[neighbor]) {
                            q.push(neighbor);
                        }
                    }
                }
            }
            currentCluster++;
        }
    }

    // Step 4: Handle border points that are not noise
    for (int i = 0; i < n; ++i) {
        if (isNoise[i] && clusterAssignments[i] != -1) {
            isNoise[i] = false; // This point is actually a border point
        }
    }

    // Step 5: Adjust number of clusters if needed
    int numClusters = currentCluster;
    if (targetClusters > 0 && numClusters != targetClusters) {
        std::cout << "[DBSCAN] Adjusting clusters from " << numClusters << " to " << targetClusters << std::endl;
        
        if (numClusters > targetClusters) {
            // Merge clusters: use hierarchical clustering on cluster centroids
            std::vector<std::vector<int>> clusterPoints;
            std::vector<int> clusterLabels;
            
            for (int c = 0; c < numClusters; ++c) {
                std::vector<int> centroid(data[0].size(), 0);
                int count = 0;
                for (int i = 0; i < n; ++i) {
                    if (clusterAssignments[i] == c) {
                        for (size_t d = 0; d < data[i].size(); ++d) {
                            centroid[d] += data[i][d];
                        }
                        count++;
                    }
                }
                if (count > 0) {
                    for (size_t d = 0; d < centroid.size(); ++d) {
                        centroid[d] /= count;
                    }
                    clusterPoints.push_back(centroid);
                    clusterLabels.push_back(c);
                }
            }
            
            std::vector<std::pair<int, int>> mergedLabels = hierarchicalClustering(clusterPoints, targetClusters);
            
            // Update cluster assignments
            for (int i = 0; i < n; ++i) {
                if (clusterAssignments[i] != -1) {
                    auto it = std::find(clusterLabels.begin(), clusterLabels.end(), clusterAssignments[i]);
                    if (it != clusterLabels.end()) {
                        int idx = std::distance(clusterLabels.begin(), it);
                        clusterAssignments[i] = mergedLabels[idx].second;
                    }
                }
            }
        } else if (numClusters < targetClusters) {
            // Split largest clusters
            std::vector<std::pair<int, int>> clusterSizes; // (size, cluster_id)
            for (int c = 0; c < numClusters; ++c) {
                int size = 0;
                for (int i = 0; i < n; ++i) {
                    if (clusterAssignments[i] == c) size++;
                }
                clusterSizes.push_back({size, c});
            }
            std::sort(clusterSizes.rbegin(), clusterSizes.rend());
            
            int needed = targetClusters - numClusters;
            int nextCluster = numClusters;
            
            for (int k = 0; k < needed && k < (int)clusterSizes.size(); ++k) {
                int clusterToSplit = clusterSizes[k].second;
                std::vector<std::vector<int>> clusterData;
                std::vector<int> originalIndices;
                
                for (int i = 0; i < n; ++i) {
                    if (clusterAssignments[i] == clusterToSplit) {
                        clusterData.push_back(data[i]);
                        originalIndices.push_back(i);
                    }
                }
                
                if (clusterData.size() >= 2) {
                    std::vector<std::pair<int, int>> splitLabels = kmeans(clusterData, 2);
                    for (size_t i = 0; i < originalIndices.size(); ++i) {
                        if (splitLabels[i].second == 1) {
                            clusterAssignments[originalIndices[i]] = nextCluster;
                        }
                    }
                    nextCluster++;
                }
            }
        }
    }

    // Step 6: Prepare result
    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < n; ++i) {
        if (clusterAssignments[i] != -1) {
            result.emplace_back(i, clusterAssignments[i]);
        }
    }

    int finalClusters = 0;
    if (!result.empty()) {
        std::set<int> uniqueClusters;
        for (const auto& pair : result) {
            uniqueClusters.insert(pair.second);
        }
        finalClusters = uniqueClusters.size();
    }

    std::cout << "[DBSCAN] Clustering complete with " << finalClusters << " clusters and " 
              << result.size() << " assigned points out of " << n << " total points." << std::endl;
    
    return result;
}

// Mean Shift clustering function (good for finding natural clusters)
vector<pair<int, int>> meanShift(const vector<vector<int>>& data, float bandwidth) {
    int n = data.size();
    int dim = data[0].size();
    vector<vector<float>> centroids(n, vector<float>(dim));
    
    // Initialize centroids as data points
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            centroids[i][j] = (float)data[i][j];
        }
    }
    
    // Mean shift iterations
    int maxIterations = 50;
    for (int iter = 0; iter < maxIterations; iter++) {
        bool converged = true;
        
        for (int i = 0; i < n; i++) {
            vector<float> newCentroid(dim, 0.0);
            float totalWeight = 0.0;
            
            // Calculate weighted mean of nearby points
            for (int j = 0; j < n; j++) {
                float dist = 0.0;
                for (int d = 0; d < dim; d++) {
                    dist += pow(centroids[i][d] - data[j][d], 2);
                }
                dist = sqrt(dist);
                
                if (dist <= bandwidth) {
                    float weight = 1.0 / (1.0 + dist / bandwidth);  // Gaussian kernel
                    for (int d = 0; d < dim; d++) {
                        newCentroid[d] += weight * data[j][d];
                    }
                    totalWeight += weight;
                }
            }
            
            // Update centroid
            if (totalWeight > 0) {
                for (int d = 0; d < dim; d++) {
                    newCentroid[d] /= totalWeight;
                }
                
                // Check convergence
                float centroidShift = 0.0;
                for (int d = 0; d < dim; d++) {
                    centroidShift += pow(newCentroid[d] - centroids[i][d], 2);
                }
                centroidShift = sqrt(centroidShift);
                
                if (centroidShift > 0.01) {
                    converged = false;
                }
                
                centroids[i] = newCentroid;
            }
        }
        
        if (converged) break;
    }
    
    // Assign cluster labels based on proximity of final centroids
    vector<int> clusterAssignments(n, -1);
    int clusterId = 0;
    float mergeThreshold = bandwidth * 0.5;
    
    for (int i = 0; i < n; i++) {
        if (clusterAssignments[i] != -1) continue;
        
        clusterAssignments[i] = clusterId;
        
        // Find points with similar centroids
        for (int j = i + 1; j < n; j++) {
            if (clusterAssignments[j] == -1) {
                float dist = 0.0;
                for (int d = 0; d < dim; d++) {
                    dist += pow(centroids[i][d] - centroids[j][d], 2);
                }
                dist = sqrt(dist);
                
                if (dist <= mergeThreshold) {
                    clusterAssignments[j] = clusterId;
                }
            }
        }
        clusterId++;
    }
    
    cout << "Mean Shift found " << clusterId << " clusters" << endl;
    
    // Convert to pair format
    vector<pair<int, int>> result;
    for (int i = 0; i < n; i++) {
        result.emplace_back(i, clusterAssignments[i]);
    }
    
    return result;
}

void displayClusterInfo(const vector<int>& clusterAssignments, int numClusters) {
    cout << "\n=== K-Means Clustering Results ===" << endl;
    cout << "Number of clusters: " << numClusters << endl;
    
    // Count points in each cluster
    vector<int> clusterSizes(numClusters, 0);
    for (int assignment : clusterAssignments) {
        clusterSizes[assignment]++;
    }
    
    cout << "\nCluster sizes:" << endl;
    for (int i = 0; i < numClusters; i++) {
        cout << "Cluster " << i << ": " << clusterSizes[i] << " points" << endl;
    }
    
    // Show first 10 assignments as example
    cout << "\nFirst 10 cluster assignments:" << endl;
    for (int i = 0; i < min(10, (int)clusterAssignments.size()); i++) {
        cout << "Point " << i << " -> Cluster " << clusterAssignments[i] << endl;
    }
    
    if (clusterAssignments.size() > 10) {
        cout << "..." << endl;
    }
}

// Grid-based data reduction - groups nearby points into grid cells
vector<vector<int>> gridBasedReduction(const vector<vector<int>>& data, int gridSize) {
    cout << "Grid-based reduction with grid size: " << gridSize << endl;
    
    if (data.empty()) return {};
    
    // Find min and max values for each dimension
    vector<int> minVals(data[0].size(), data[0][0]);
    vector<int> maxVals(data[0].size(), data[0][0]);
    
    for (const auto& point : data) {
        for (size_t i = 0; i < point.size(); i++) {
            minVals[i] = min(minVals[i], point[i]);
            maxVals[i] = max(maxVals[i], point[i]);
        }
    }
    
    // Create grid and assign points to cells
    map<vector<int>, vector<int>> grid; // grid coordinates -> point indices
    
    for (size_t i = 0; i < data.size(); i++) {
        vector<int> gridCoords;
        for (size_t j = 0; j < data[i].size(); j++) {
            int coord = (data[i][j] - minVals[j]) / gridSize;
            gridCoords.push_back(coord);
        }
        grid[gridCoords].push_back(i);
    }
    
    // Create representative points for each grid cell
    vector<vector<int>> reducedData;
    for (const auto& cell : grid) {
        if (cell.second.empty()) continue;
        
        // Calculate centroid of the cell
        vector<int> centroid(data[0].size(), 0);
        for (int idx : cell.second) {
            for (size_t j = 0; j < data[idx].size(); j++) {
                centroid[j] += data[idx][j];
            }
        }
        
        for (size_t j = 0; j < centroid.size(); j++) {
            centroid[j] /= cell.second.size();
        }
        
        reducedData.push_back(centroid);
    }
    
    cout << "Reduced from " << data.size() << " to " << reducedData.size() << " points" << endl;
    return reducedData;
}

// Random sampling - randomly selects a subset of points
vector<vector<int>> randomSampling(const vector<vector<int>>& data, int sampleSize) {
    cout << "Random sampling to " << sampleSize << " points" << endl;
    
    if (data.size() <= sampleSize) return data;
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, data.size() - 1);
    
    set<int> selectedIndices;
    while (selectedIndices.size() < sampleSize) {
        selectedIndices.insert(dis(gen));
    }
    
    vector<vector<int>> sampledData;
    for (int idx : selectedIndices) {
        sampledData.push_back(data[idx]);
    }
    
    cout << "Sampled " << sampledData.size() << " points from " << data.size() << " total" << endl;
    return sampledData;
}

// K-means based reduction - uses k-means to find representative points
vector<vector<int>> kmeansReduction(const vector<vector<int>>& data, int reductionFactor) {
    cout << "K-means based reduction with factor: " << reductionFactor << endl;
    
    int targetClusters = data.size() / reductionFactor;
    if (targetClusters < 1) targetClusters = 1;
    
    // Use k-means to find cluster centroids
    vector<pair<int, int>> clusterAssignments = kmeans(data, targetClusters);
    
    // Calculate centroids
    vector<vector<int>> centroids(targetClusters, vector<int>(data[0].size(), 0));
    vector<int> clusterSizes(targetClusters, 0);
    
    for (const auto& assignment : clusterAssignments) {
        int cluster = assignment.second;
        for (size_t j = 0; j < data[assignment.first].size(); j++) {
            centroids[cluster][j] += data[assignment.first][j];
        }
        clusterSizes[cluster]++;
    }
    
    // Average the centroids
    for (int i = 0; i < targetClusters; i++) {
        if (clusterSizes[i] > 0) {
            for (size_t j = 0; j < centroids[i].size(); j++) {
                centroids[i][j] /= clusterSizes[i];
            }
        }
    }
    
    cout << "Reduced from " << data.size() << " to " << centroids.size() << " centroids" << endl;
    return centroids;
}

// Density-based reduction - removes points in dense regions
vector<vector<int>> densityBasedReduction(const vector<vector<int>>& data, float eps, int minPts) {
    cout << "Density-based reduction with eps: " << eps << ", minPts: " << minPts << endl;
    
    vector<bool> keepPoint(data.size(), true);
    
    // Calculate point densities
    for (size_t i = 0; i < data.size(); i++) {
        int neighbors = 0;
        for (size_t j = 0; j < data.size(); j++) {
            if (i != j && calculateDistance(data[i], data[j]) <= eps) {
                neighbors++;
            }
        }
        
        // If point has many neighbors, mark it for removal (keep only sparse points)
        if (neighbors >= minPts) {
            keepPoint[i] = false;
        }
    }
    
    // Collect kept points
    vector<vector<int>> reducedData;
    for (size_t i = 0; i < data.size(); i++) {
        if (keepPoint[i]) {
            reducedData.push_back(data[i]);
        }
    }
    
    cout << "Reduced from " << data.size() << " to " << reducedData.size() << " points" << endl;
    return reducedData;
} 