#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include "mutual_info.hpp" // あなたのヘッダーをインクルード

// Pythonのnp.linspaceのような等差数列を生成する関数
std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> vec;
    if (num == 0) return vec;
    if (num == 1) {
        vec.push_back(start);
        return vec;
    }
    double delta = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        vec.push_back(start + delta * i);
    }
    return vec;
}

// 相関のある2変量正規分布データを生成する関数
void generate_multivariate_normal(
    std::vector<double>& x_data,
    std::vector<double>& y_data,
    int N,
    double mean_x, double mean_y,
    double var_x, double var_y,
    double cov_xy
) {
    x_data.resize(N);
    y_data.resize(N);

    // コレステキー分解
    double l11 = std::sqrt(var_x);
    double l21 = cov_xy / l11;
    double l22 = std::sqrt(var_y - l21 * l21);

    // 乱数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 1);

    for (int i = 0; i < N; ++i) {
        double z1 = dist(gen);
        double z2 = dist(gen);
        x_data[i] = mean_x + l11 * z1;
        y_data[i] = mean_y + l21 * z1 + l22 * z2;
    }
}

int main() {
    // Pythonスクリプトのパラメータを定義
    const double var_x = 1.0;
    const double cov = 1.0;
    const int N = 10000;
    const int num_points = 50;
    
    std::vector<double> Var_y = linspace(1.01, 10.0, num_points);
    std::vector<double> MIt(num_points), MI5(num_points), MI10(num_points), MI50(num_points);

    std::cout << "計算を開始します..." << std::endl;

    // Pythonのループに相当する処理
    for (int i = 0; i < num_points; ++i) {
        double var_y = Var_y[i];
        
        // 1. 理論値を計算
        double rho = cov / std::sqrt(var_x * var_y);
        MIt[i] = -0.5 * std::log(1.0 - rho * rho);

        // 2. 相関のあるガウス分布データを生成
        std::vector<double> x_vec, y_vec;
        generate_multivariate_normal(x_vec, y_vec, N, 0.0, 0.0, var_x, var_y, cov);

        // 3. C++関数に渡すためのポインタ準備（Pythonラッパーの役割）
        double* x_ptr = x_vec.data();
        double* y_ptr = y_vec.data();
        double** x_ptr_ptr = &x_ptr;
        double** y_ptr_ptr = &y_ptr;

        // 4. 相互情報量を計算
        MI5[i] = mutual_info<double>(x_ptr_ptr, y_ptr_ptr, 5, 1, 1, N);
        MI10[i] = mutual_info<double>(x_ptr_ptr, y_ptr_ptr, 10, 1, 1, N);
        MI50[i] = mutual_info<double>(x_ptr_ptr, y_ptr_ptr, 50, 1, 1, N);
        
        std::cout << "進捗: " << i + 1 << "/" << num_points << " (Var_y=" << var_y << ")" << std::endl;
    }
    
    std::cout << "計算が完了しました。results.csv に結果を出力します。" << std::endl;

    // 5. 結果をCSVファイルに出力
    std::ofstream outfile("results.csv");
    outfile << "Var_y,Theoretical_MI,MI_k5,MI_k10,MI_k50\n";
    for (int i = 0; i < num_points; ++i) {
        outfile << std::fixed << std::setprecision(8) 
                << Var_y[i] << "," << MIt[i] << "," 
                << MI5[i] << "," << MI10[i] << "," << MI50[i] << "\n";
    }
    outfile.close();

    std::cout << "ファイル出力が完了しました。\n";
    std::cout << "次のコマンドでグラフを生成してください: gnuplot plot.gp" << std::endl;

    return 0;
}
