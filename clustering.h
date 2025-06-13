#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <map>
#include <set>
#include <queue>

using namespace std;

// K-means clustering
vector<pair<int, int>> kmeans(const vector<vector<int>>& data, int k);

// Hierarchical clustering
vector<pair<int, int>> hierarchicalClustering(const vector<vector<int>>& data, int k);

// DBSCAN clustering
vector<pair<int, int>> dbscan(const vector<vector<int>>& data, float eps, int minPts, int targetClusters);

// Mean shift clustering
vector<pair<int, int>> meanShift(const vector<vector<int>>& data, float bandwidth);

// Preliminary clustering algorithms for data reduction
vector<vector<int>> gridBasedReduction(const vector<vector<int>>& data, int gridSize);
vector<vector<int>> randomSampling(const vector<vector<int>>& data, int sampleSize);
vector<vector<int>> kmeansReduction(const vector<vector<int>>& data, int reductionFactor);
vector<vector<int>> densityBasedReduction(const vector<vector<int>>& data, float eps, int minPts);

// Utility functions
void displayClusterInfo(const vector<int>& clusterAssignments, int numClusters);
float calculateDistance(const vector<int>& point1, const vector<int>& point2);

#endif 