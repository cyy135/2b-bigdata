#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>
#include <queue>
#include <climits>
#include "clustering.h"
using namespace std;

// Global variable to store the number of dimensions (columns)
int dimension = 0;

void compare_dimensions(const vector<vector<int>>& data, int dim1, int dim2) {
    cout << "\n=== Dimension Comparison Graph ===" << endl;
    cout << "Comparing dimension " << dim1 << " vs dimension " << dim2 << endl;
    cout << "Data points: " << data.size() << endl;
    
    // Find min and max values for both dimensions
    int min1 = data[0][dim1], max1 = data[0][dim1];
    int min2 = data[0][dim2], max2 = data[0][dim2];
    
    for (const auto& row : data) {
        min1 = min(min1, row[dim1]);
        max1 = max(max1, row[dim1]);
        min2 = min(min2, row[dim2]);
        max2 = max(max2, row[dim2]);
    }
    
    cout << "Dimension " << dim1 << " range: [" << min1 << ", " << max1 << "]" << endl;
    cout << "Dimension " << dim2 << " range: [" << min2 << ", " << max2 << "]" << endl;
    
    // Calculate image dimensions based on actual data range (scaled down)
    int scaleFactor = 10;  // Make image 10x smaller
    int width = max(1, (max1 - min1 + 1) / scaleFactor);
    int height = max(1, (max2 - min2 + 1) / scaleFactor);
    
    // Create PGM image
    vector<vector<int>> image(height, vector<int>(width, 0));
    
    // Plot data points
    for (const auto& row : data) {
        int x = (row[dim1] - min1) / scaleFactor;
        int y = (row[dim2] - min2) / scaleFactor;
        
        // Ensure coordinates are within bounds
        if (x >= 0 && x < width && y >= 0 && y < height) {
            // Flip y coordinate for PGM (origin at top-left)
            y = height - 1 - y;
            // Add 50 to pixel value (cumulative effect)
            image[y][x] = min(255, image[y][x] + 50);
        }
    }
    
    // Write PGM file
    string filename = "dimension_" + to_string(dim1) + "_vs_" + to_string(dim2) + ".pgm";
    ofstream pgmFile(filename);
    
    if (pgmFile.is_open()) {
        pgmFile << "P2" << endl;  // PGM format
        pgmFile << width << " " << height << endl;  // Width and height
        pgmFile << "255" << endl;  // Max value
        
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                pgmFile << image[i][j];
                if (j < width - 1) pgmFile << " ";
            }
            pgmFile << endl;
        }
        
        pgmFile.close();
        cout << "PGM file created: " << filename << endl;
        cout << "Image size: " << width << "x" << height << " pixels" << endl;
    } else {
        cerr << "Error: Could not create PGM file" << endl;
    }
}

vector<vector<int>> data_input(char* path) {
    vector<vector<int>> data;
    ifstream file(path);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open file '" << path << "'" << endl;
        return data;
    }

    string line;
    bool firstLine = true;
    
    while (getline(file, line)) {
        if (firstLine) {
            // First line determines the dimension (number of columns)
            stringstream ss(line);
            string cell;
            dimension = 0;
            
            while (getline(ss, cell, ',')) {
                dimension++;
            }
            
            // First column is index, so subtract 1 from dimension
            dimension = dimension - 1;
            
            cout << "Detected dimension: " << dimension << " columns (excluding index column)" << endl;
            firstLine = false;
            continue; // Skip the header row for data
        }
        
        vector<int> row;
        stringstream ss(line);
        string cell;
        bool firstColumn = true;

        // Parse CSV line (handles basic CSV format)
        while (getline(ss, cell, ',')) {
            // Skip the first column (index column)
            if (firstColumn) {
                firstColumn = false;
                continue;
            }
            
            // Remove quotes if present
            if (cell.length() >= 2 && cell.front() == '"' && cell.back() == '"') {
                cell = cell.substr(1, cell.length() - 2);
            }
            
            // Convert string to int (truncate decimal part)
            try {
                // First try to convert to float, then truncate to int
                float floatValue = stof(cell);
                int value = (int)floatValue;  // Truncate decimal part
                row.push_back(value);
            } catch (const invalid_argument& e) {
                // If conversion fails, skip this cell or use 0
                cerr << "Warning: Could not convert '" << cell << "' to int, using 0" << endl;
                row.push_back(0);
            }
        }
        data.push_back(row);
    }

    file.close();
    cout << "Successfully loaded " << data.size() << " rows of data." << endl;
    cout << "Global dimension variable: " << dimension << endl;
    
    // Display first few rows
    cout << "\nFirst 5 data:" << endl;
    for (size_t i = 0; i < min((size_t)5, data.size()); ++i) {
        cout << "Data " << (i + 1) << ": ";
        for (size_t j = 0; j < data[i].size(); ++j) {
            cout << data[i][j];
            if (j < data[i].size() - 1) cout << ", ";
        }
        cout << endl;
    }
    return data;
}

// Fast cluster assignment using centroids only (less accurate but very fast)
vector<int> fastClusterAssignment(const vector<vector<int>>& originalData, 
                                 const vector<pair<int, int>>& clusterAssignments,
                                 const vector<vector<int>>& clusterData) {
    int n = originalData.size();
    vector<int> assignments(n, -1);
    
    // Compute cluster centroids
    unordered_map<int, vector<float>> centroids;
    unordered_map<int, int> clusterSizes;
    
    for (const auto& assignment : clusterAssignments) {
        int clusterId = assignment.second;
        int pointIdx = assignment.first;
        
        if (centroids.find(clusterId) == centroids.end()) {
            centroids[clusterId] = vector<float>(clusterData[pointIdx].size(), 0.0f);
            clusterSizes[clusterId] = 0;
        }
        
        for (size_t d = 0; d < clusterData[pointIdx].size(); ++d) {
            centroids[clusterId][d] += clusterData[pointIdx][d];
        }
        clusterSizes[clusterId]++;
    }
    
    // Normalize centroids
    for (auto& [clusterId, centroid] : centroids) {
        for (size_t d = 0; d < centroid.size(); ++d) {
            centroid[d] /= clusterSizes[clusterId];
        }
    }
    
    cout << "Fast centroid-based assignment using " << centroids.size() << " clusters..." << endl;
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < n; ++i) {
        float minDist = numeric_limits<float>::max();
        int bestCluster = -1;
        
        for (const auto& [clusterId, centroid] : centroids) {
            float dist = 0.0f;
            for (size_t d = 0; d < originalData[i].size(); ++d) {
                float diff = originalData[i][d] - centroid[d];
                dist += diff * diff;
            }
            dist = sqrt(dist);
            
            if (dist < minDist) {
                minDist = dist;
                bestCluster = clusterId;
            }
        }
        
        assignments[i] = bestCluster;
    }
    
    cout << "Fast cluster assignment completed for " << n << " points" << endl;
    return assignments;
}

// Accurate and fast cluster assignment using representative points and spatial indexing
vector<int> accurateClusterAssignment(const vector<vector<int>>& originalData, 
                                     const vector<pair<int, int>>& clusterAssignments,
                                     const vector<vector<int>>& clusterData) {
    int n = originalData.size();
    vector<int> assignments(n, -1);
    
    // Step 1: Extract representative points for each cluster (not just centroids)
    unordered_map<int, vector<int>> clusterPoints;
    unordered_map<int, vector<float>> centroids;
    unordered_map<int, vector<float>> minBounds, maxBounds;
    
    for (const auto& assignment : clusterAssignments) {
        int clusterId = assignment.second;
        int pointIdx = assignment.first;
        clusterPoints[clusterId].push_back(pointIdx);
    }
    
    // Compute centroids and bounding boxes for spatial indexing
    for (const auto& [clusterId, points] : clusterPoints) {
        int dim = clusterData[points[0]].size();
        centroids[clusterId] = vector<float>(dim, 0.0f);
        minBounds[clusterId] = vector<float>(dim, numeric_limits<float>::max());
        maxBounds[clusterId] = vector<float>(dim, numeric_limits<float>::lowest());
        
        for (int pointIdx : points) {
            for (size_t d = 0; d < clusterData[pointIdx].size(); ++d) {
                float val = clusterData[pointIdx][d];
                centroids[clusterId][d] += val;
                minBounds[clusterId][d] = min(minBounds[clusterId][d], val);
                maxBounds[clusterId][d] = max(maxBounds[clusterId][d], val);
            }
        }
        
        // Normalize centroid
        for (size_t d = 0; d < centroids[clusterId].size(); ++d) {
            centroids[clusterId][d] /= points.size();
        }
    }
    
    // Step 2: Two-phase assignment for accuracy
    cout << "Accurate cluster assignment using " << clusterPoints.size() << " clusters..." << endl;
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < n; ++i) {
        float minDist = numeric_limits<float>::max();
        int bestCluster = -1;
        
        // Phase 1: Quick filtering using centroids and bounding boxes
        vector<int> candidateClusters;
        for (const auto& [clusterId, centroid] : centroids) {
            const auto& minBound = minBounds[clusterId];
            const auto& maxBound = maxBounds[clusterId];
            
            // Quick centroid distance check
            float centroidDist = 0.0f;
            for (size_t d = 0; d < originalData[i].size(); ++d) {
                float diff = originalData[i][d] - centroid[d];
                centroidDist += diff * diff;
            }
            centroidDist = sqrt(centroidDist);
            
            // If centroid is too far, skip this cluster
            if (centroidDist > minDist * 2.0f) continue;
            
            // Bounding box check
            bool outsideBox = false;
            for (size_t d = 0; d < originalData[i].size(); ++d) {
                if (originalData[i][d] < minBound[d] - minDist * 1.5f || 
                    originalData[i][d] > maxBound[d] + minDist * 1.5f) {
                    outsideBox = true;
                    break;
                }
            }
            if (outsideBox) continue;
            
            candidateClusters.push_back(clusterId);
        }
        
        // Phase 2: Accurate assignment using actual cluster points
        for (int clusterId : candidateClusters) {
            const auto& points = clusterPoints[clusterId];
            
            // Find minimum distance to any point in this cluster
            for (int pointIdx : points) {
                float dist = 0.0f;
                bool earlyTerminate = false;
                
                for (size_t d = 0; d < originalData[i].size(); ++d) {
                    float diff = originalData[i][d] - clusterData[pointIdx][d];
                    dist += diff * diff;
                    
                    // Early termination
                    if (dist >= minDist * minDist) {
                        earlyTerminate = true;
                        break;
                    }
                }
                
                if (!earlyTerminate) {
                    dist = sqrt(dist);
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = clusterId;
                    }
                }
            }
        }
        
        assignments[i] = bestCluster;
    }
    
    cout << "Accurate cluster assignment completed for " << n << " points" << endl;
    return assignments;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <csv_file_path>" << endl;
        cout << "  csv_file_path: Path to the input CSV file" << endl;
        cout << "  Note: Filtering automatically discards 5% of data with 10% axis threshold" << endl;
        return 1;
    }

    vector<vector<int>> originalData = data_input(argv[1]);
    
    if (originalData.empty()) {
        cout << "No data loaded or file is empty." << endl;
        return 1;
    }

    // Use original data values directly without transformation
    vector<vector<int>> data = originalData;
    int numDimensions = data[0].size();

    // Filter out points with distance to origin > 1500 or sum > 4000
    vector<vector<int>> filteredData;
    for (const auto& row : data) {
        float distToOrigin = 0.0f;
        int sum = 0;
        for (int val : row) {
            distToOrigin += val * val;
            sum += val;
        }
        distToOrigin = sqrt(distToOrigin);
        if (distToOrigin <= 1700 && sum < dimension*400) {
            filteredData.push_back(row);
        }
    }
    cout << "Filtered out " << (data.size() - filteredData.size()) 
         << " points with distance to origin > 1700 or sum too big" << endl;

    // Use filtered data for sampling
    data = randomSampling(filteredData, filteredData.size()/5);


    // Set DBSCAN parameters for your data range and dimensions
    float eps = 50;
    int minPts = 8;
    cout << "\nDBSCAN clustering (all dimensions)" << endl;
    cout << "Using epsilon (distance threshold): " << eps << endl;
    cout << "Using minimum points for a cluster: " << minPts << endl;

    // Perform DBSCAN clustering on original data
    vector<pair<int, int>> clusterAssignments = dbscan(data, eps, minPts, dimension*4-1);
    
    // No back-transformation needed since we're using original values
    vector<vector<int>> backTransformedData = data;

    // Choose assignment method: true for accurate, false for fast
    bool useAccurateAssignment = true;  // Set to false for maximum speed

    vector<int> originalAssignments;
    if (useAccurateAssignment) {
        cout << "Using ACCURATE assignment method (slower but more precise)" << endl;
        originalAssignments = accurateClusterAssignment(originalData, clusterAssignments, backTransformedData);
    } else {
        cout << "Using FAST assignment method (faster but less precise)" << endl;
        originalAssignments = fastClusterAssignment(originalData, clusterAssignments, backTransformedData);
    }

    // Generate output filename based on input filename pattern
    string inputFile = argv[1];
    string outputFile;
    
    // Check if input file ends with "_data.csv" and replace with "_submission.csv"
    if (inputFile.length() >= 9 && inputFile.substr(inputFile.length() - 9) == "_data.csv") {
        outputFile = inputFile.substr(0, inputFile.length() - 9) + "_submission.csv";
    } else {
        // Fallback: just replace .csv with _submission.csv
        size_t dotPos = inputFile.find_last_of('.');
        if (dotPos != string::npos) {
            outputFile = inputFile.substr(0, dotPos) + "_submission.csv";
        } else {
            outputFile = inputFile + "_submission.csv";
        }
    }
    
    cout << "Input file: " << inputFile << endl;
    cout << "Output file: " << outputFile << endl;

    ofstream csvFile(outputFile);
    csvFile << "id,label\n";
    for (size_t i = 0; i < originalAssignments.size(); ++i) {
        csvFile << (i + 1) << "," << originalAssignments[i] << "\n";
    }
    csvFile.close();
    cout << "Cluster assignments for all original data written to " << outputFile << std::endl;

    return 0;
}