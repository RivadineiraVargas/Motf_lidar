#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <array>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


// Definição das dimensões e constantes
#define DIM1 64
#define DIM2 2650
#define DIM3_INTENSITY 1
#define DIM3_COLOR 3
#define POINTS_PER_RECORD 4
#define POSE_VALUES 4
#define BBOX_POINTS 8

// Estruturas de dados
typedef struct {
    double x;
    double y;
    double z;
} astro_vector_3D_t;

typedef struct {
    astro_vector_3D_t p0, p1, p2, p3, p4, p5, p6, p7;
} t_3d_bbox_struct;

typedef struct {
    int b;
    int g;
    int r;
    double cartesian_x;
    double cartesian_y;
    double cartesian_z;
    double range;
} lidar_point;


bool read_bin_file(
    const char* path,
    size_t &num_points,
    cv::Mat &out_image,
    float meters,
    int size_in_pixels,
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr
);



// Declarações das funções utilitárias

float max(float arr[], int size);
float min(float arr[], int size);
int point_in_polygon(float px, float py, float base_points[4][2]);
void color_points_within_bbox(std::vector<lidar_point> &lidar_points, std::vector<std::vector<float>> all_transformed_bbox_for_rangeview);
cv::Mat normalizar(const cv::Mat& range_image, const cv::Mat& colors_image, float meters);
int extract_number_from_filename(const char *filename);
int compare_files(const void *a, const void *b);
std::vector<std::string> list_subdirectories(const char *dir_path);
std::vector<std::string> list_files_with_extension(const char *dir_path, const std::string &extension);
std::string take_substring(int position, const std::string& separator, const std::string& name);
void multiply_matrix_vector(const float mat[4][4], const float vec[4], float result[4]);
int invert_matrix(const float mat[4][4], float inv[4][4]);
void transform_vertices(const std::vector<std::array<float, 3>>& vertices, float lidar_pose[4][4], std::vector<float>& result);
void transformar_para_sistema_lidar_topo(const std::vector<std::array<float, 3>>& bbox, float lidar_pose[4][4], std::vector<float>& result);
void draw_bounding_box_birdview(const std::vector<float>& result, cv::Mat& birdview_image, float meters, int scale);
void calculate_birdview_image(cv::Mat &birdview_image_color, size_t num_points, float meters, int size_in_pixels);
void read_bin_file(const char *filename, size_t &num_points, cv::Mat &birdview_image, float meters, int size_in_pixels);
bool read_pose_file(const char *filename, float pose[4][4]);
void read_bbox_file(const char *objs_bbox_dir, const std::string& scene_name, const std::string& pose_name, std::vector<std::vector<std::array<float, 3>>> &all_bbox, float lidar_pose[4][4], cv::Mat& birdview_image, std::vector<std::vector<float>> &all_transformed_bbox_for_rangeview, float meters, int size_in_pixels, bool show_bboxes);


#endif // UTILS_HPP
