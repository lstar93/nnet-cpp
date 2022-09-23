#include "plot.h"

namespace plotter {

    point_t get_random_point(const std::pair<int,int>& x_range, const std::pair<int,int>& y_range) {
        std::random_device rd; // obtain random device
        std::mt19937 mt19(rd()); // seed the generator
        std::uniform_real_distribution<> urdX(x_range.first, x_range.second);
        std::uniform_real_distribution<> urdY(y_range.first, y_range.second);

        return {urdX(mt19), urdY(mt19)};
    }

    std::vector<point_t> get_random_points(const std::pair<int,int>& x_range, const std::pair<int,int>& y_range, size_t size) {
        std::random_device rd; // obtain random device
        std::mt19937 mt19(rd()); // seed the generator
        std::uniform_real_distribution<> urdX(x_range.first, x_range.second);
        std::uniform_real_distribution<> urdY(y_range.first, y_range.second);

        std::vector<point_t> points;
        for(auto i=0; i<size; ++i) {
            points.push_back({urdX(mt19), urdY(mt19)});
        }

        return points;
    }

    // plot group of points
    void plotter::plot_points(const std::vector<point_t>& points, const double marker_size, const keywords_t& keywords) {
        if(type == plottype::separate) {
            plt::figure();
        }

        std::vector<double> x;
        std::vector<double> y;

        for(auto& p: points) {
            x.push_back(p.x);
            y.push_back(p.y);
        }

        plt::scatter(x, y, marker_size, keywords);
    }

    // plot several groups of points
    void plotter::plot_points(const std::vector<std::vector<point_t>>& points_groups, const double marker_size, const keywords_t& keywords) {
        for(auto& points: points_groups) {
            plot_points(points, marker_size, keywords);
        }
    }

    // plot points divided by a function
    void plotter::plot_points(const std::vector<point_t>& points, const std::function<double(double)>& linear_function, const double marker_size, const keywords_t& keywords) {
        std::vector<point_t> points1, points2;
        for(auto& p: points) {
            if(linear_function(p.x) <= p.y) {
                points1.push_back(p);
            } else {
                points2.push_back(p);
            }
        }
        plot_points({points1, points2}, marker_size, keywords);
    }

    void plotter::plot_function(const std::function<double(double)>& linear_function, const std::pair<int,int>& xlimits, const std::pair<int,int>& ylimits, const labels_t& labels, const double points_number) {
        if(type == plottype::separate) {
            plt::figure();
        }
        
        std::vector<double> x;
        std::vector<double> y;

        for(auto i = -points_number/2; i < points_number/2; ++i) {
            x.push_back(i);
            y.push_back(linear_function(i));
        }

        // do not limit plot for all limits eq to 0
        if(xlimits.first == xlimits.second == ylimits.first == ylimits.second == 0) {
            plt::xlim(xlimits.first, xlimits.second);
            plt::ylim(ylimits.first, ylimits.second);
        }

        if(!labels.x.empty()) {
            plt::xlabel(labels.x);
        }

        if(!labels.y.empty()) {
            plt::ylabel(labels.y);
        }

        plt::plot(x, y);
    }

    void plotter::show() {
        plt::legend();
        plt::show();
    }
}