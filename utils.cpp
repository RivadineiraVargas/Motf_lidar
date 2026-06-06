#include "utils.hpp"
#include <float.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

using namespace std;

// Função para calcular o valor máximo de uma coordenada Z
float max(float arr[], int size) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

// Função para calcular o valor mínimo de uma coordenada Z
float min(float arr[], int size) {
    float min_val = FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    return min_val;
}

// Função para verificar se um ponto está dentro de um polígono (algoritmo de ray-casting)
int point_in_polygon(float px, float py, float base_points[4][2]) {
    int i, j, c = 0;
    for (i = 0, j = 3; i < 4; j = i++) {
        if (((base_points[i][1] > py) != (base_points[j][1] > py)) &&
            (px < (base_points[j][0] - base_points[i][0]) * (py - base_points[i][1]) / (base_points[j][1] - base_points[i][1]) + base_points[i][0])) {
            c = !c;
        }
    }
    return c;
}

// Função para processar os vértices das caixas delimitadoras e colorir os pontos LIDAR
void color_points_within_bbox(vector<lidar_point> &lidar_points, vector<vector<float>> all_transformed_bbox_for_rangeview) {
    float* bbox_max_height = (float*)malloc(all_transformed_bbox_for_rangeview.size() * sizeof(float));
    float* bbox_min_height = (float*)malloc(all_transformed_bbox_for_rangeview.size() * sizeof(float));

    for (int i = 0; i < (int) all_transformed_bbox_for_rangeview.size(); i++) {
        float base_points[4][2];
        for (int j = 0; j < 4; j++) {
            base_points[j][0] = all_transformed_bbox_for_rangeview[i][j * 3];
            base_points[j][1] = all_transformed_bbox_for_rangeview[i][j * 3 + 1];
        }

        float z_values[8];
        for (int j = 0; j < 8; j++) {
            z_values[j] = all_transformed_bbox_for_rangeview[i][j * 3 + 2];
        }

        bbox_max_height[i] = max(z_values, 8);
        bbox_min_height[i] = min(z_values, 8);

        for (int k = 0; k < (int) lidar_points.size(); k++) {
            float px = lidar_points[k].cartesian_x;
            float py = lidar_points[k].cartesian_y;
            float pz = lidar_points[k].cartesian_z;

            if (point_in_polygon(px, py, base_points) && pz <= bbox_max_height[i]) {
                lidar_points[k].r = 1.0f;
                lidar_points[k].g = 0.0f;
                lidar_points[k].b = 0.0f;
            }
        }
    }

    free(bbox_max_height);
    free(bbox_min_height);
}

// Função de normalização
cv::Mat normalizar(const cv::Mat& range_image, const cv::Mat& colors_image, float meters) {
    if (range_image.rows != colors_image.rows || range_image.cols != colors_image.cols) {
        cerr << "As dimensões de range_image e colors_image não correspondem." << endl;
        return cv::Mat();
    }

    int lines = range_image.rows;
    int columns = range_image.cols;
    float resolution = (meters * 100.0f) / 256.0f;
    cv::Mat image_normalized = cv::Mat::zeros(lines, columns, CV_8UC1);

    for(int x = 0; x < lines; ++x) {
        for(int y = 0; y < columns; ++y) {
            float pixel = range_image.at<float>(x, y);
            if(pixel < 0.0f) {
                image_normalized.at<uchar>(x, y) = 0;
            }
            else {
                int value = static_cast<int>(255.0f - ((pixel * 100.0f) / resolution));
                value = std::min(std::max(value, 0), 255);
                image_normalized.at<uchar>(x, y) = static_cast<uchar>(value);
            }
        }
    }

    cv::Mat image_color;
    cv::cvtColor(image_normalized, image_color, cv::COLOR_GRAY2BGR);
    cv::Mat mask = range_image <= -2.0f;

    for(int x = 0; x < lines; ++x) {
        for(int y = 0; y < columns; ++y) {
            cv::Vec3f color = colors_image.at<cv::Vec3f>(x, y);
            bool isRed = (abs(color[0] - 1.0f) < 1e-3) && (abs(color[1] - 0.0f) < 1e-3) && (abs(color[2] - 0.0f) < 1e-3);

            if(isRed || mask.at<uchar>(x, y)) {
                image_color.at<cv::Vec3b>(x, y) = cv::Vec3b(0, 0, 255);
            }
        }
    }

    return image_color;
}

// Função para extrair o número do nome do arquivo
int extract_number_from_filename(const char *filename) {
    const char *ptr = strrchr(filename, '/');
    if (ptr) {
        filename = ptr + 1;
    }
    int num = 0;
    while (*filename >= '0' && *filename <= '9') {
        num = num * 10 + (*filename - '0');
        filename++;
    }
    return num;
}

// Função para comparar dois arquivos
int compare_files(const void *a, const void *b) {
    const char *file_a = *(const char**)a;
    const char *file_b = *(const char**)b;
    int num_a = extract_number_from_filename(file_a);
    int num_b = extract_number_from_filename(file_b);
    return num_a - num_b;
}

// Função para listar subdiretórios
vector<string> list_subdirectories(const char *dir_path) {
    vector<string> subdirs;
    DIR *dp = opendir(dir_path);
    if (dp == NULL) {
        perror("Erro ao abrir o diretório");
        return subdirs;
    }

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        string entry_name = string(entry->d_name);
        if (entry_name == "." || entry_name == "..") continue;

        string full_path = string(dir_path) + "/" + entry_name;

        struct stat path_stat;
        if (stat(full_path.c_str(), &path_stat) != 0) {
            perror("Erro ao obter informações do subdiretório");
            continue;
        }

        if (S_ISDIR(path_stat.st_mode)) {
            subdirs.push_back(entry_name);
        }
    }

    closedir(dp);
    return subdirs;
}

// Função para listar arquivos com uma extensão específica
vector<string> list_files_with_extension(const char *dir_path, const string &extension) {
    vector<string> files;
    DIR *dp = opendir(dir_path);
    if (dp == NULL) {
        perror("Erro ao abrir o diretório");
        return files;
    }

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        string entry_name = string(entry->d_name);
        if (entry_name == "." || entry_name == "..") continue;

        string full_path = string(dir_path) + "/" + entry_name;

        struct stat path_stat;
        if (stat(full_path.c_str(), &path_stat) != 0) {
            perror("Erro ao obter informações do arquivo");
            continue;
        }

        if (S_ISREG(path_stat.st_mode)) {
            if (entry_name.size() >= extension.size()) {
                if (entry_name.compare(entry_name.size() - extension.size(), extension.size(), extension) == 0) {
                    files.push_back(entry_name);
                }
            }
        }
    }

    closedir(dp);
    return files;
}

// Função para extrair uma substring
string take_substring(int position, const string& separator, const string& name) {
    vector<string> tokens;
    size_t start = 0, end;

    while ((end = name.find(separator, start)) != string::npos) {
        tokens.push_back(name.substr(start, end - start));
        start = end + separator.length();
    }
    tokens.push_back(name.substr(start));

    if (position >= 1 && position <= (int) tokens.size()) {
        return tokens[position - 1];
    } else {
        return "";
    }
}

// Funções de matriz
void multiply_matrix_vector(const float mat[4][4], const float vec[4], float result[4]) {
    for (int i = 0; i < 4; i++) {
        result[i] = 0.0f;
        for (int j = 0; j < 4; j++) {
            result[i] += mat[i][j] * vec[j];
        }
    }
}

int invert_matrix(const float mat[4][4], float inv[4][4]) {
    cv::Mat mat_cv(4, 4, CV_32F);
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            mat_cv.at<float>(i,j) = mat[i][j];

    cv::Mat inv_cv;
    bool success = cv::invert(mat_cv, inv_cv, cv::DECOMP_SVD);
    if (!success) {
        return 0;
    }

    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            inv[i][j] = inv_cv.at<float>(i,j);

    return 1;
}

void transform_vertices(const vector<array<float, 3>>& vertices, float lidar_pose[4][4], vector<float>& result) {
    float lidar_pose_inverse[4][4];
    if (!invert_matrix(lidar_pose, lidar_pose_inverse)) {
        cerr << "Erro ao inverter a matriz de pose." << endl;
        return;
    }

    for (size_t i = 0; i < vertices.size(); i++) {
        float homogeneous_vertex[4] = { vertices[i][0], vertices[i][1], vertices[i][2], 1.0f };
        float transformed_homogeneous[4];

        multiply_matrix_vector(lidar_pose_inverse, homogeneous_vertex, transformed_homogeneous);
        result.push_back(transformed_homogeneous[0]);
        result.push_back(transformed_homogeneous[1]);
        result.push_back(transformed_homogeneous[2]);
    }
}

void transformar_para_sistema_lidar_topo(const vector<array<float, 3>>& bbox, float lidar_pose[4][4], vector<float>& result) {
    float lidar_pose_inversa[4][4];
    
    if (!invert_matrix(lidar_pose, lidar_pose_inversa)) {
        printf("Erro ao inverter a matriz de pose.\n");
        return;
    }

    for (size_t i = 0; i < bbox.size(); i++) {
        float track_coords_homogeneas[4] = { bbox[i][0], bbox[i][1], bbox[i][2], 1.0f };
        float track_coords_transformadas[4];
        
        multiply_matrix_vector(lidar_pose_inversa, track_coords_homogeneas, track_coords_transformadas);

        result.push_back(track_coords_transformadas[0]);
        result.push_back(track_coords_transformadas[1]);
        result.push_back(track_coords_transformadas[2]);
    }
}

// Funções de visualização
void draw_bounding_box_birdview(const vector<float>& result, cv::Mat& birdview_image, float meters, int scale) {
    if (result.size() < 24) {
        printf("Bounding box incompleto para desenho.\n");
        return;
    }

    int edges[12][2] = {
        {0,1}, {1,2}, {2,3}, {3,0},
        {4,5}, {5,6}, {6,7}, {7,4},
        {0,4}, {1,5}, {2,6}, {3,7}
    };

    for(int i = 0; i < 12; i++) {
        int idx1 = edges[i][0];
        int idx2 = edges[i][1];

        float x1 = result[idx1*3];
        float y1 = result[idx1*3 +1];
        float x2 = result[idx2*3];
        float y2 = result[idx2*3 +1];

        if (x1 > -meters && x1 < meters && y1 > -meters && y1 < meters &&
            x2 > -meters && x2 < meters && y2 > -meters && y2 < meters) {
            int x_pixel1 = static_cast<int>((x1 + meters) * scale);
            int y_pixel1 = static_cast<int>((meters - y1) * scale);
            int x_pixel2 = static_cast<int>((x2 + meters) * scale);
            int y_pixel2 = static_cast<int>((meters - y2) * scale);

            cv::line(birdview_image, 
                        cv::Point(x_pixel1, y_pixel1), 
                        cv::Point(x_pixel2, y_pixel2), 
                        cv::Scalar(0, 255, 0), 2);
        }
    }
}

void calculate_birdview_image(cv::Mat &birdview_image_color, size_t num_points, float meters, int size_in_pixels) {
    int scale = size_in_pixels / (2 * meters);
    float *points_xyz = (float*) malloc(num_points * POINTS_PER_RECORD * sizeof(float));
    
    cv::Mat birdview_image = cv::Mat::zeros(size_in_pixels, size_in_pixels, CV_8UC1);

    for (size_t i = 0; i < num_points; i++) {
        float x = points_xyz[i * POINTS_PER_RECORD];
        float y = points_xyz[i * POINTS_PER_RECORD + 1];
        if (-meters < x && x < meters && -meters < y && y < meters) {
            int x_pixel = static_cast<int>((x + meters) * scale);
            int y_pixel = static_cast<int>((meters - y) * scale);

            if (x_pixel >= 0 && x_pixel < size_in_pixels && y_pixel >= 0 && y_pixel < size_in_pixels) {
                birdview_image.at<uchar>(y_pixel, x_pixel) = 255; 
            }
        }
    }
    cv::cvtColor(birdview_image, birdview_image_color, cv::COLOR_GRAY2BGR);
    free(points_xyz);
}



bool read_bin_file(
    const char* path,
    size_t &num_points,
    cv::Mat &out_image,
    float meters,
    int size_in_pixels,
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr
) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Falha ao abrir arquivo .bin: " << path << "\n";
        return false;
    }

    // Cada registro é (x, y, z, intensidade)
    ifs.seekg(0, std::ios::end);
    size_t num_bytes = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    num_points = num_bytes / (4 * sizeof(float));
    if (cloud_ptr) {
        cloud_ptr->clear();
        cloud_ptr->points.reserve(num_points);
    }

    out_image = cv::Mat::zeros(size_in_pixels, size_in_pixels, CV_8UC1);

    const float scale = size_in_pixels / (meters * 2.0f);

    for (size_t i = 0; i < num_points; i++) {
        float data[4];
        ifs.read(reinterpret_cast<char*>(&data), sizeof(float) * 4);
        float x = data[0];
        float y = data[1];
        float z = data[2];
        float intensity = data[3];

        // Preenche PointCloud (se fornecido)
        if (cloud_ptr) {
            pcl::PointXYZI p;
            p.x = x;
            p.y = y;
            p.z = z;
            p.intensity = intensity;
            cloud_ptr->points.push_back(p);
        }

        // Preenche Birdview
        int px = int((x + meters) * scale);
        int py = int((y + meters) * scale);

        if (px >= 0 && px < size_in_pixels && py >= 0 && py < size_in_pixels) {
            out_image.at<uchar>(size_in_pixels - 1 - py, px) =
                std::min(255, int(intensity * 255));
        }
    }

    if (cloud_ptr) {
        cloud_ptr->width = cloud_ptr->points.size();
        cloud_ptr->height = 1;
        cloud_ptr->is_dense = false;
    }

    return true;
}


// Funções de leitura de arquivo
/*void read_bin_file(const char *filename, size_t &num_points, cv::Mat &birdview_image, float meters, int size_in_pixels) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Erro ao abrir o arquivo");
        return;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size % (POINTS_PER_RECORD * sizeof(float)) != 0) {
        fprintf(stderr, "O arquivo binário tem um tamanho inválido: %s\n", filename);
        fclose(file);
        return;
    }

    num_points = file_size / (POINTS_PER_RECORD * sizeof(float));
    float *points = (float*) malloc(num_points * POINTS_PER_RECORD * sizeof(float));
    if (points == NULL) {
        perror("Erro ao alocar memória");
        fclose(file);
        return;
    }

    size_t read_count = fread(points, sizeof(float), num_points * POINTS_PER_RECORD, file);
    if (read_count != num_points * POINTS_PER_RECORD) {
        fprintf(stderr, "Erro ao ler o arquivo binário: %s\n", filename);
        free(points);
        fclose(file);
        return;
    }

    free(points);
    fclose(file);
}
*/
bool read_pose_file(const char *filename, float pose[4][4]) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Erro ao abrir o arquivo de pose");
        return false;
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (fscanf(file, "%f", &pose[i][j]) != 1) {
                fprintf(stderr, "Erro ao ler pose no arquivo: %s\n", filename);
                fclose(file);
                return false;
            }
        }
    }

    fclose(file);
    return true;
}

void read_bbox_file(const char *objs_bbox_dir, const string& scene_name, const string& pose_name, vector<vector<array<float, 3>>> &all_bbox, float lidar_pose[4][4], cv::Mat& birdview_image, vector<vector<float>> &all_transformed_bbox_for_rangeview, float meters, int size_in_pixels, bool show_bboxes) {
    int scale = size_in_pixels / (2 * meters);
    string pose_number = take_substring(1, ".", pose_name);
    if (pose_number.empty()) {
        printf("Pose number extraído está vazio para pose_name: %s\n", pose_name.c_str());
        return;
    }

    string bbox_dir = string(objs_bbox_dir) + "/" + scene_name + "/" + pose_number;
    DIR *dp = opendir(bbox_dir.c_str());
    if (dp == NULL) {
        perror("Erro ao abrir o diretório de bbox");
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        string entry_name = string(entry->d_name);
        if (entry_name == "." || entry_name == "..") continue;

        string full_path = bbox_dir + "/" + entry_name;

        struct stat path_stat;
        if (stat(full_path.c_str(), &path_stat) != 0) {
            perror("Erro ao obter informações do arquivo de bbox");
            continue;
        }

        if (S_ISREG(path_stat.st_mode) && entry_name.find(".txt") != string::npos) {
            FILE *file = fopen(full_path.c_str(), "r");
            if (file == NULL) {
                perror("Erro ao abrir o arquivo de bbox");
                continue;
            }

            vector<array<float, 3>> bbox_aux;
            bool read_success = true;
            for (int i = 0; i < 8; i++) {
                array<float, 3> p;
                if (fscanf(file, "%f %f %f", &p[0], &p[1], &p[2]) != 3) {
                    fprintf(stderr, "Erro ao ler bbox no arquivo: %s\n", full_path.c_str());
                    read_success = false;
                    break;
                }
                bbox_aux.push_back(p);
            }

            if (read_success) {
                all_bbox.push_back(bbox_aux);
                vector<float> transformed_bbox_for_birdeyeview;
                vector<float> transformed_bbox_for_rangeview;
                transformar_para_sistema_lidar_topo(bbox_aux, lidar_pose, transformed_bbox_for_birdeyeview);
                transform_vertices(bbox_aux, lidar_pose, transformed_bbox_for_rangeview);
                all_transformed_bbox_for_rangeview.push_back(transformed_bbox_for_rangeview);
                if (show_bboxes) {
                    draw_bounding_box_birdview(transformed_bbox_for_birdeyeview, birdview_image, meters, scale);
                }
            }

            fclose(file);
        }
    }

    closedir(dp);
}
