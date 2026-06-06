#include "utils.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <opencv/highgui.h>

namespace fs = std::filesystem;
using namespace std;

// ── Predições do modelo (exportadas por export_predictions_global.py) ──────────
// kind: 0 = histórico (cinza), 1 = futuro real (verde), 2 = futuro predito (vermelho)
struct TrajPoint { int kind; int t; float x, y, z; };
// scene -> obj_id -> pontos (coords GLOBAIS)
static std::map<std::string, std::map<std::string, std::vector<TrajPoint>>> g_predictions;

static void load_predictions(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        printf("Aviso: predições não encontradas em '%s' (viewer roda sem elas).\n", path.c_str());
        return;
    }
    std::string scene, oid, line;
    int kind, t; float x, y, z, count = 0;
    while (std::getline(ifs, line)) {
        std::istringstream ss(line);
        if (ss >> scene >> oid >> kind >> t >> x >> y >> z) {
            g_predictions[scene][oid].push_back({kind, t, x, y, z});
            count++;
        }
    }
    printf("Predições carregadas: %.0f pontos.\n", count);
}

// Desenha as trajetórias no BEV: transforma global->sensor com inv(pose) e projeta.
static void draw_predictions_birdview(cv::Mat& birdview, const std::string& scene,
                                      float pose[4][4], float meters, int size_in_pixels) {
    auto it = g_predictions.find(scene);
    if (it == g_predictions.end()) return;

    float scale = size_in_pixels / (2.0f * meters);
    float inv[4][4];
    if (!invert_matrix(pose, inv)) return;

    for (auto& kv : it->second) {
        for (int kind = 0; kind <= 2; ++kind) {
            std::vector<std::pair<int, cv::Point>> proj;
            for (auto& p : kv.second) {
                if (p.kind != kind) continue;
                float gh[4] = { p.x, p.y, p.z, 1.0f }, s[4];
                multiply_matrix_vector(inv, gh, s);
                int xp = (int)((s[0] + meters) * scale);
                int yp = (int)((meters - s[1]) * scale);
                proj.push_back({ p.t, cv::Point(xp, yp) });
            }
            std::sort(proj.begin(), proj.end(),
                      [](const std::pair<int,cv::Point>& a, const std::pair<int,cv::Point>& b){
                          return a.first < b.first; });
            cv::Scalar color = (kind == 0) ? cv::Scalar(160,160,160)   // histórico
                             : (kind == 1) ? cv::Scalar(0,255,0)       // real (verde)
                                           : cv::Scalar(0,0,255);      // predito (vermelho)
            int thick = (kind == 0) ? 1 : 2;
            for (size_t i = 1; i < proj.size(); ++i)
                cv::line(birdview, proj[i-1].second, proj[i].second, color, thick, cv::LINE_AA);
            for (auto& pr : proj)
                cv::circle(birdview, pr.second, 3, color, -1, cv::LINE_AA);
        }
    }
}

int main(int argc, char** argv) {
    bool draw_red_points = true;
    bool show = true;
    std::string input_base_path = "";
    int wait_delay = 1; // Padrão: 1ms (máxima velocidade)
    float meters = 51.2f;
    int size_in_pixels = 1024;
    bool paused = false;
    bool show_bboxes = false;
    bool show_predictions = true;

    // --- 1. Processamento de Argumentos ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-no_red") {
            draw_red_points = false;
        } else if (arg == "-no_show") {
            show = false;
        } else if (arg == "--input") {
            if (i + 1 < argc) {
                input_base_path = argv[++i];
            } else {
                std::cerr << "Erro: O argumento --input requer um caminho." << std::endl;
                return EXIT_FAILURE;
            }
        } else if (arg == "-v") {
            if (i + 1 < argc) {
                try {
                    wait_delay = std::stoi(argv[++i]);
                    if (wait_delay < 1) wait_delay = 1;
                } catch (...) {
                    std::cerr << "Erro: Valor inválido para -v. Use um número inteiro (ex: -v 100)." << std::endl;
                    return EXIT_FAILURE;
                }
            } else {
                std::cerr << "Erro: O argumento -v requer um valor em milissegundos." << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    // --- 2. Validação do Caminho de Entrada ---
    if (input_base_path.empty()) {
        std::cerr << "Erro: Caminho de entrada nao especificado." << std::endl;
        std::cerr << "Uso: " << argv[0] << " --input <caminho> [-v <ms>] [-no_red] [-no_show]" << std::endl;
        return EXIT_FAILURE;
    }

    if (input_base_path.back() == '/') {
        input_base_path.pop_back();
    }

    std::string bin_root_str = input_base_path + "/bin_files";
    std::string pose_root_str = input_base_path + "/poses";
    std::string bbox_root_str = input_base_path + "/objs_bbox";
    std::string images_root_str = input_base_path + "/images";
 
    const char *bin_root_dir = bin_root_str.c_str();
    const char *pose_root_dir = pose_root_str.c_str();
    const char *bbox_root_dir = bbox_root_str.c_str();

    std::chrono::duration<double> global_delta_time(0.0);

    // Carregar predições do modelo (se existir o arquivo no diretório atual)
    load_predictions("predictions_global.txt");

    // --- 3. Listar Cenas ---
    vector<string> scenes = list_subdirectories(bin_root_dir);
    if (scenes.empty()) {
        printf("Nenhuma cena encontrada em: %s\n", bin_root_dir);
        return EXIT_FAILURE;
    }

    std::vector<float> intensities(DIM1 * DIM2 * DIM3_INTENSITY, 0.0f);
    std::vector<float> all_points_colors(DIM1 * DIM2 * DIM3_COLOR, 0.0f);

    int index = 1;

    // --- 4. Loop das Cenas ---
    for (int s=0; s < scenes.size(); ++s) {
        auto initial_time = std::chrono::high_resolution_clock::now();
        printf("\nProcessando cena: %s\n", scenes[s].c_str());

        string bin_scene_dir = string(bin_root_dir) + "/" + scenes[s];
        string pose_scene_dir = string(pose_root_dir) + "/" + scenes[s];
        string bbox_scene_dir = string(bbox_root_dir) + "/" + scenes[s];
        string images_scene_dir = string(images_root_str) + "/" + scenes[s];
        fs::path images_dir_path = images_scene_dir;

        if (!fs::exists(images_dir_path)) {
            if (fs::create_directories(images_dir_path)) {
                std::cout << "Pastas criadas com sucesso: " << images_dir_path << "\n";
            } else {
                std::cerr << "Erro ao criar pastas (ou elas já existiam).\n";
            }
        }
        struct stat st;
        if (stat(pose_scene_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            printf("Diretório de poses não encontrado: %s\n", scenes[s].c_str());
            continue;
        }
        if (stat(bbox_scene_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            printf("Diretório de bbox não encontrado: %s\n", scenes[s].c_str());
            continue;
        }

        vector<string> bin_files = list_files_with_extension(bin_scene_dir.c_str(), ".bin");
        vector<string> pose_files = list_files_with_extension(pose_scene_dir.c_str(), ".txt");

        if (bin_files.empty() || pose_files.empty()) {
            printf("Arquivos faltando na cena: %s\n", scenes[s].c_str());
            continue;
        }

        size_t num_files = min(bin_files.size(), pose_files.size());

        const char **bin_files_c = (const char **)malloc(bin_files.size() * sizeof(char *));
        for (size_t i = 0; i < bin_files.size(); i++) {
            string full_path = bin_scene_dir + "/" + bin_files[i];
            bin_files_c[i] = strdup(full_path.c_str());
        }

        const char **pose_files_c = (const char **)malloc(pose_files.size() * sizeof(char *));
        for (size_t i = 0; i < pose_files.size(); i++) {
            string full_path = pose_scene_dir + "/" + pose_files[i];
            pose_files_c[i] = strdup(full_path.c_str());
        }

        qsort((void*)bin_files_c, bin_files.size(), sizeof(char*), compare_files);
        qsort((void*)pose_files_c, pose_files.size(), sizeof(char*), compare_files);

        const std::string winName = "LiDAR Range and Bird's-Eye View";
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::resizeWindow(winName, 1280, 720);
        // cv::setWindowProperty(winName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
                
        // --- 5. Loop dos Arquivos ---
        int i = 0;
        while (i < num_files && i >= 0) {
            auto local_initial_time = std::chrono::high_resolution_clock::now();

            string bin_file_path = string(bin_files_c[i]);
            string pose_file_path = string(pose_files_c[i]);
            cv::Mat birdview_image;
            char strIndex[12];
            snprintf(strIndex, sizeof(strIndex), "%d", i);

            string image_file_path = string(images_scene_dir) + "/" + strIndex + ".png";
            size_t num_points;
            read_bin_file(bin_file_path.c_str(), num_points, birdview_image, meters, size_in_pixels);
            if (!fs::exists(image_file_path)) {
                calculate_birdview_image(birdview_image, num_points, meters, size_in_pixels);

                if (cv::imwrite(image_file_path, birdview_image)) {
                    std::cout << "Imagem criada com sucesso: " << image_file_path << "\n";
                } else {
                    std::cerr << "Erro ao criar imagens (ou elas já existiam).\n";
                }
            }
            else 
            {
                birdview_image = cv::imread(image_file_path);
                if (!birdview_image.empty()) {
                    std::cout << "Imagem já existia e foi carreagada com sucesso: " << image_file_path << "\n";
                }
            }

            float pose[4][4];
            if (!read_pose_file(pose_file_path.c_str(), pose)) {
                printf("Falha ao ler pose: %s\n", pose_file_path.c_str());
                continue;
            }

            size_t last_slash = pose_file_path.find_last_of('/');
            string filename_full = (last_slash == string::npos) ? pose_file_path : pose_file_path.substr(last_slash + 1);
            size_t last_dot = filename_full.find_last_of('.');
            string pose_file_name = (last_dot == string::npos) ? filename_full : filename_full.substr(0, last_dot);
            
            if (pose_file_name.empty()) continue;

            vector<vector<array<float, 3>>> all_bbox;
            vector<vector<float>> all_transformed_bbox_for_rangeview;
            read_bbox_file(bbox_root_dir, scenes[s], pose_file_name, all_bbox, pose, birdview_image, all_transformed_bbox_for_rangeview, meters, size_in_pixels, show_bboxes);

            // Desenhar trajetórias preditas/reais sobre o BEV (tecla 't' alterna)
            if (show_predictions)
                draw_predictions_birdview(birdview_image, scenes[s], pose, meters, size_in_pixels);

            if (show) {
                float *points_xyz = (float*) malloc(num_points * POINTS_PER_RECORD * sizeof(float));

                // --- CORREÇÃO: LEITURA DOS DADOS ---
                FILE *stream = fopen(bin_file_path.c_str(), "rb");
                if (stream) {
                    size_t num_read = fread(points_xyz, sizeof(float), num_points * POINTS_PER_RECORD, stream);
                    fclose(stream);
                    
                    // Validação opcional para garantir que leu tudo
                    if (num_read != num_points * POINTS_PER_RECORD) {
                        std::cerr << "Aviso: Leitura incompleta do arquivo binário." << std::endl;
                    }
                } else {
                    std::cerr << "Erro: Nao foi possivel abrir o arquivo binario para leitura 3D." << std::endl;
                }
                // ------------------------------------

                vector<lidar_point> lidar_points;
                for (size_t k = 0; k < num_points; k++) {
                    lidar_point point;
                    point.cartesian_x   = (double) points_xyz[k * POINTS_PER_RECORD];    
                    point.cartesian_y   = (double) points_xyz[k * POINTS_PER_RECORD + 1];
                    point.cartesian_z   = (double) points_xyz[k * POINTS_PER_RECORD + 2];
                    point.range         = (double) points_xyz[k * POINTS_PER_RECORD + 3];
                    point.r             = 0.0;
                    point.g             = 0.0;
                    point.b             = 1.0;

                    lidar_points.push_back(point);
                    intensities[k] = lidar_points[k].range;
                }

                if (draw_red_points)
                    color_points_within_bbox(lidar_points, all_transformed_bbox_for_rangeview);
                
                size_t max_points = std::min((size_t)(DIM1 * DIM2), lidar_points.size());
                for (size_t k = 0; k < max_points; ++k) {
                    all_points_colors[k * 3 + 0] = (float)lidar_points[k].r;
                    all_points_colors[k * 3 + 1] = (float)lidar_points[k].g;
                    all_points_colors[k * 3 + 2] = (float)lidar_points[k].b;
                }

                cv::Mat range_image_reshaped(DIM1, DIM2, CV_32F, intensities.data());
                cv::Mat all_points_colors_reshaped(DIM1, DIM2, CV_32FC3, all_points_colors.data());

                float threshold = 75.0f;
                cv::Mat normalized_image = normalizar(range_image_reshaped, all_points_colors_reshaped, threshold);

                cv::Mat resized_range;
                cv::resize(normalized_image, resized_range, cv::Size(1920, 128), 0, 0, cv::INTER_LINEAR);

                cv::Mat resized_birdview;
                cv::resize(birdview_image, resized_birdview, cv::Size(920, 920), 0, 0, cv::INTER_LINEAR);

                cv::Mat bg(1080, 1920, CV_8UC3, cv::Scalar(255, 255, 255));
                resized_range.copyTo(bg(cv::Rect(0, 0, resized_range.cols, resized_range.rows)));

                cv::Rect roi_birdview(480, 128, resized_birdview.cols, resized_birdview.rows);
                resized_birdview.copyTo(bg(roi_birdview));

                cv::imshow(winName, bg);

                while (true) {
                    int key = cv::waitKey(wait_delay);

                    if (key == 32) {
                        paused = !paused; // Espaço
                    }
                    else if ((key == 97) || (key == 65)) {
                        i -= 1;
                        if (i < 0) s -= 2;
                        break; // a
                    } 
                    else if ((key == 68) || (key == 100)) {
                        i += 1;
                        break; // d
                    }
                    else if ((key == 81) || (key == 113)) {
                        i = -1;
                        s -= 2;
                        break; // q
                    }     
                    else if ((key == 69) || (key == 101)) {
                        i = num_files;
                        break; // e
                    }   
                    else if ((key == 66) || (key == 98)) {
                        show_bboxes = !show_bboxes;
                        break; // b
                    }
                    else if ((key == 84) || (key == 116)) {
                        show_predictions = !show_predictions;
                        break; // t — alterna trajetórias preditas
                    }
                    else if ((key == 82) || (key == 114)) {
                        draw_red_points = !draw_red_points;
                        break; // r
                    }   
                    else if (key == 27) {
                        return 0;    // ESC
                    }
                    else if ((key == 83) || (key == 115)) {
                        FILE *file;
                        char filename[256]; 
                        snprintf(filename, sizeof(filename), "%s/%s.txt", input_base_path.c_str(), "trash");
                        printf("%s\n", filename);
                        file = fopen(filename, "a");

                        if (file == NULL) {
                            printf("Erro ao abrir ou criar o arquivo de lista de cenas!\n");
                            return 1;
                        }

                        fprintf(file, scenes[s].c_str());
                        fprintf(file, "\n");

                        // Fecha o arquivo para salvar as alterações
                        fclose(file);

                        printf("Linha adicionada com sucesso.\n");
                        
                        
                    }
                    if (!paused) {
                        i++;
                        break;
                    }
                }

                lidar_points.clear();
                free(points_xyz);

            }

            all_bbox.clear();

            auto local_final_time = std::chrono::high_resolution_clock::now();
            global_delta_time += (local_final_time - local_initial_time);
            index++;
            if (i < 0) {
                break;
            }
        }

        for (size_t i = 0; i < bin_files.size(); i++) free((void*)bin_files_c[i]);
        free(bin_files_c);
        for (size_t i = 0; i < pose_files.size(); i++) free((void*)pose_files_c[i]);
        free(pose_files_c);
        if (s < -1){
            s = -1;
        }
    }
    
    if (index > 1) index--; 
    std::cout << " Global Deltatime Average: " << global_delta_time.count()/index << " seconds" << std::endl;

    return EXIT_SUCCESS;
}