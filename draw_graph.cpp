#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

void ensure_directory(const string& dir) {
    struct stat st = {0};
    if (stat(dir.c_str(), &st) == -1) {
        mkdir(dir.c_str(), 0755);
    }
}

vector<vector<int>> read_csv(const string& path, int& dimension) {
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
            stringstream ss(line);
            string cell;
            dimension = 0;
            while (getline(ss, cell, ',')) dimension++;
            dimension -= 1;
            firstLine = false;
            continue;
        }
        vector<int> row;
        stringstream ss(line);
        string cell;
        bool firstColumn = true;
        while (getline(ss, cell, ',')) {
            if (firstColumn) { firstColumn = false; continue; }
            try {
                float f = stof(cell);
                row.push_back((int)f);
            } catch (...) { row.push_back(0); }
        }
        data.push_back(row);
    }
    return data;
}

struct RGB {
    unsigned char r, g, b;
    RGB(unsigned char r=255, unsigned char g=255, unsigned char b=255) : r(r), g(g), b(b) {}
};

// Minimal 5x7 bitmap font for digits and minus sign
const int FONT_W = 5, FONT_H = 7;
const unsigned char font[11][FONT_H] = {
    {0x1E,0x29,0x25,0x23,0x21,0x29,0x1E}, // 0
    {0x08,0x18,0x08,0x08,0x08,0x08,0x1C}, // 1
    {0x1E,0x21,0x01,0x0E,0x10,0x20,0x3F}, // 2
    {0x1E,0x21,0x01,0x0E,0x01,0x21,0x1E}, // 3
    {0x02,0x06,0x0A,0x12,0x3F,0x02,0x02}, // 4
    {0x3F,0x20,0x3E,0x01,0x01,0x21,0x1E}, // 5
    {0x0E,0x10,0x20,0x3E,0x21,0x21,0x1E}, // 6
    {0x3F,0x01,0x02,0x04,0x08,0x08,0x08}, // 7
    {0x1E,0x21,0x21,0x1E,0x21,0x21,0x1E}, // 8
    {0x1E,0x21,0x21,0x1F,0x01,0x02,0x1C}, // 9
    {0x00,0x00,0x00,0x0E,0x00,0x00,0x00}  // -
};

void draw_char(vector<vector<RGB>>& image, int x, int y, char c, RGB color) {
    int idx;
    if (c >= '0' && c <= '9') idx = c - '0';
    else if (c == '-') idx = 10;
    else return;
    for (int row = 0; row < FONT_H; ++row) {
        for (int col = 0; col < FONT_W; ++col) {
            if (font[idx][row] & (1 << (4-col))) {
                int px = x + col, py = y + row;
                if (py >= 0 && py < (int)image.size() && px >= 0 && px < (int)image[0].size())
                    image[py][px] = color;
            }
        }
    }
}

void draw_string(vector<vector<RGB>>& image, int x, int y, const string& s, RGB color) {
    for (size_t i = 0; i < s.size(); ++i) {
        draw_char(image, x + i * (FONT_W + 1), y, s[i], color);
    }
}

// HSV to RGB conversion
void hsv2rgb(float h, float s, float v, int &r, int &g, int &b) {
    float c = v * s;
    float x = c * (1 - fabs(fmod(h / 60.0, 2) - 1));
    float m = v - c;
    float r1, g1, b1;
    if (h < 60)      { r1 = c; g1 = x; b1 = 0; }
    else if (h < 120){ r1 = x; g1 = c; b1 = 0; }
    else if (h < 180){ r1 = 0; g1 = c; b1 = x; }
    else if (h < 240){ r1 = 0; g1 = x; b1 = c; }
    else if (h < 300){ r1 = x; g1 = 0; b1 = c; }
    else             { r1 = c; g1 = 0; b1 = x; }
    r = (int)((r1 + m) * 255);
    g = (int)((g1 + m) * 255);
    b = (int)((b1 + m) * 255);
}

void draw_ppm(const vector<vector<int>>& data, int dim1, int dim2, const string& outdir) {
    int min1 = 0, max1 = data[0][dim1];
    int min2 = 0, max2 = data[0][dim2];
    for (const auto& row : data) {
        max1 = max(max1, row[dim1]);
        max2 = max(max2, row[dim2]);
    }
    int scaleFactor = 5;
    int width = max(1, (max1 - min1 + 1) / scaleFactor);
    int height = max(1, (max2 - min2 + 1) / scaleFactor);
    int margin = 40;
    int imgw = width + margin + 1;
    int imgh = height + margin + 1;
    vector<vector<RGB>> image(imgh, vector<RGB>(imgw, RGB(255,255,255)));
    // Compute pixel position for (0,0) in data space
    int origin_x = margin + (int)round((0 - min1) / (double)scaleFactor);
    int origin_y = height - (int)round((0 - min2) / (double)scaleFactor);
    // Draw axes at the origin
    if (origin_x >= margin && origin_x < imgw) {
        for (int y = 0; y < height; ++y) image[y][origin_x] = RGB(0,0,0); // y-axis
    }
    if (origin_y >= 0 && origin_y < imgh) {
        for (int x = margin; x < imgw; ++x) image[origin_y][x] = RGB(0,0,0); // x-axis
    }
    // Draw ticks (black) every 100 pixels in the image, relative to the new axes
    int tickPixelInterval = 100;
    // X axis ticks
    for (int x = margin; x < imgw; x += tickPixelInterval) {
        if (origin_y >= 0 && origin_y < imgh) {
            for (int dy = 0; dy < 5; ++dy) image[origin_y+dy][x] = RGB(0,0,0);
        }
        int tick_value = min1 + (int)((x - margin) * scaleFactor);
        cout << "  X tick: value=" << tick_value << ", pixel=" << x << endl;
        draw_string(image, x - FONT_W, origin_y + 8, to_string(tick_value), RGB(0,0,0));
    }
    // Y axis ticks
    for (int y = height; y >= 0; y -= tickPixelInterval) {
        if (origin_x >= margin && origin_x < imgw) {
            for (int dx = 0; dx < 5; ++dx) image[y][origin_x-dx] = RGB(0,0,0);
        }
        int tick_value = min2 + (int)((height - y) * scaleFactor);
        cout << "  Y tick: value=" << tick_value << ", pixel=" << y << endl;
        draw_string(image, origin_x - 35, y - FONT_H/2, to_string(tick_value), RGB(0,0,0));
    }
    // Count hits for each pixel
    vector<vector<int>> hit_count(imgh, vector<int>(imgw, 0));
    for (const auto& row : data) {
        int x = (row[dim1] - min1) / scaleFactor + margin;
        int y = height - (row[dim2] - min2) / scaleFactor;
        if (x >= margin && x < imgw && y >= 0 && y < imgh) {
            hit_count[y][x]++;
        }
    }
    // Set color based on hit count using HSV color wheel
    for (int y = 0; y < imgh; ++y) {
        for (int x = 0; x < imgw; ++x) {
            if (hit_count[y][x] > 0) {
                float hue = fmod(hit_count[y][x] * 8.0f, 360.0f); // 8 deg per hit, counterclockwise
                int r, g, b;
                hsv2rgb(hue, 1.0f, 1.0f, r, g, b);
                image[y][x] = RGB(r, g, b);
            }
        }
    }
    // Save as PPM
    string filename = outdir + "/dimension_" + to_string(dim1) + "_vs_" + to_string(dim2) + ".ppm";
    ofstream ppmFile(filename, ios::binary);
    ppmFile << "P6\n" << imgw << " " << imgh << "\n255\n";
    for (int y = 0; y < imgh; ++y) {
        for (int x = 0; x < imgw; ++x) {
            ppmFile << image[y][x].r << image[y][x].g << image[y][x].b;
        }
    }
    ppmFile.close();
    cout << "Saved: " << filename << endl;
    // Print axis scale info
    cout << "  X axis (dim " << dim1 << "): " << min1 << " to " << max1 << endl;
    cout << "  Y axis (dim " << dim2 << "): " << min2 << " to " << max2 << endl;
}

// Read cluster labels from *_submission.csv
vector<int> read_clusters(const string& path) {
    vector<int> clusters;
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "Error: Could not open cluster file '" << path << "'" << endl;
        return clusters;
    }
    string line;
    bool firstLine = true;
    while (getline(file, line)) {
        if (firstLine) { firstLine = false; continue; } // skip header
        stringstream ss(line);
        string cell;
        getline(ss, cell, ','); // skip index
        if (!getline(ss, cell, ',')) continue;
        try {
            clusters.push_back(stoi(cell));
        } catch (...) {
            clusters.push_back(0);
        }
    }
    return clusters;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_data.csv>" << endl;
        return 1;
    }
    string input_path = argv[1];
    string outdir = input_path;
    if (outdir.size() > 4 && outdir.substr(outdir.size() - 4) == ".csv")
        outdir = outdir.substr(0, outdir.size() - 4);
    outdir += ".result";
    ensure_directory(outdir);
    int dimension = 0;
    vector<vector<int>> data = read_csv(input_path, dimension);
    if (data.empty() || dimension < 2) {
        cerr << "Not enough data or dimensions." << endl;
        return 1;
    }
    for (int i = 0; i < dimension; ++i) {
        for (int j = i + 1; j < dimension; ++j) {
            draw_ppm(data, i, j, outdir);
        }
    }
    cout << "All PPM files saved in: " << outdir << endl;
    cout << "(You can convert .ppm to .jpg using: convert file.ppm file.jpg)" << endl;
    // After the main dimension-vs-dimension plots, add dimension-vs-cluster plots
    // Find cluster file name
    string cluster_path = input_path;
    if (cluster_path.size() > 9 && cluster_path.substr(cluster_path.size() - 9) == "_data.csv")
        cluster_path = cluster_path.substr(0, cluster_path.size() - 9) + "_submission.csv";
    else if (cluster_path.size() > 4 && cluster_path.substr(cluster_path.size() - 4) == ".csv")
        cluster_path = cluster_path.substr(0, cluster_path.size() - 4) + "_submission.csv";
    vector<int> clusters = read_clusters(cluster_path);
    if (clusters.size() != data.size()) {
        cerr << "Warning: cluster result size does not match data size! Skipping cluster comparison plots." << endl;
    } else {
        // For each pair of dimensions, plot with cluster color
        if (clusters.size() == data.size()) {
            // Find number of clusters
            int maxCluster = 0;
            for (int c : clusters) maxCluster = max(maxCluster, c);
            int numClusters = maxCluster + 1;
            // Assign a color to each cluster using HSV
            vector<RGB> cluster_colors(numClusters);
            for (int c = 0; c < numClusters; ++c) {
                int r, g, b;
                float hue = fmod((c * 360.0f) / numClusters, 360.0f);
                hsv2rgb(hue, 1.0f, 1.0f, r, g, b);
                cluster_colors[c] = RGB(r, g, b);
            }
            for (int dim_i = 0; dim_i < dimension; ++dim_i) {
                for (int dim_j = dim_i + 1; dim_j < dimension; ++dim_j) {
                    int minX = 0, maxX = data[0][dim_i];
                    int minY = 0, maxY = data[0][dim_j];
                    for (size_t i = 0; i < data.size(); ++i) {
                        maxX = max(maxX, data[i][dim_i]);
                        maxY = max(maxY, data[i][dim_j]);
                    }
                    int scaleFactorX = 5, scaleFactorY = 5;
                    int width = max(1, (maxX - minX + 1) / scaleFactorX);
                    int height = max(1, (maxY - minY + 1) / scaleFactorY);
                    int margin = 40;
                    int imgw = width + margin + 1;
                    int imgh = height + margin + 1;
                    vector<vector<RGB>> image(imgh, vector<RGB>(imgw, RGB(255,255,255)));
                    // Axes at 0
                    int origin_x = margin;
                    int origin_y = height;
                    for (int y = 0; y < height; ++y) image[y][origin_x] = RGB(0,0,0);
                    for (int x = margin; x < imgw; ++x) image[origin_y][x] = RGB(0,0,0);
                    // Ticks (every 100 pixels)
                    int tickPixelInterval = 100;
                    for (int x = margin; x < imgw; x += tickPixelInterval) {
                        for (int dy = 0; dy < 5; ++dy) image[origin_y+dy][x] = RGB(0,0,0);
                        int tick_value = minX + (int)((x - margin) * scaleFactorX);
                        draw_string(image, x - FONT_W, origin_y + 8, to_string(tick_value), RGB(0,0,0));
                    }
                    for (int y = height; y >= 0; y -= tickPixelInterval) {
                        for (int dx = 0; dx < 5; ++dx) image[y][origin_x-dx] = RGB(0,0,0);
                        int tick_value = minY + (int)((height - y) * scaleFactorY);
                        draw_string(image, origin_x - 35, y - FONT_H/2, to_string(tick_value), RGB(0,0,0));
                    }
                    // Plot points with cluster color
                    for (size_t i = 0; i < data.size(); ++i) {
                        int x = (data[i][dim_i] - minX) / scaleFactorX + margin;
                        int y = height - (data[i][dim_j] - minY) / scaleFactorY;
                        int c = clusters[i];
                        if (x >= margin && x < imgw && y >= 0 && y < imgh && c >= 0 && c < numClusters) {
                            image[y][x] = cluster_colors[c];
                        }
                    }
                    // Save
                    string filename = outdir + "/dimension_" + to_string(dim_i) + "_vs_dimension_" + to_string(dim_j) + "_cluster.ppm";
                    ofstream ppmFile(filename, ios::binary);
                    ppmFile << "P6\n" << imgw << " " << imgh << "\n255\n";
                    for (int y = 0; y < imgh; ++y) {
                        for (int x = 0; x < imgw; ++x) {
                            ppmFile << image[y][x].r << image[y][x].g << image[y][x].b;
                        }
                    }
                    ppmFile.close();
                    cout << "Saved: " << filename << endl;
                }
            }
        }
    }
    return 0;
} 