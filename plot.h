#pragma once

#define _USE_MATH_DEFINES // for sin/log

#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <utility>
#include <random>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

namespace plotter {
    using keywords_t = std::map<std::string,std::string>;

    struct point_t {
        double x, y;
    };

    point_t get_random_point(const std::pair<int,int>& x_range, const std::pair<int,int>& y_range);
    std::vector<point_t> get_random_points(const std::pair<int,int>& x_range, const std::pair<int,int>& y_range, size_t size);

    enum class plottype {
        separate,
        joint
    };

    struct labels_t {
        std::string x;
        std::string y;
    };

    class plotter {
        plotter(plottype _type): type(_type) {}
        plottype type;
    public:
        static plotter create(plottype p) {
            return std::move(plotter(p));
        }

        static void show(); // always explicitly tell to show plots or not

        void plot_points(const std::vector<point_t>& points, const double marker_size = 1.0, const keywords_t& keywords = {});
        void plot_points(const std::vector<std::vector<point_t>>& points_groups, const double marker_size = 1.0, const keywords_t& keywords = {});
        void plot_points(const std::vector<point_t>& points, const std::function<double(double)>& linear_function, const double marker_size = 1.0, const keywords_t& keywords = {});

        void plot_function(const std::function<double(double)>& linear_function, const std::pair<int,int>& xlimits = {0,0}, const std::pair<int,int>& ylimits = {0,0}, const labels_t& labels = {"",""}, const double points_number = 100);
    };
}

// Examples:

    // std::vector<plotter::point_t> points {{0,0}, {2,2}, {8,8}};
    //plotter::plot_points(points);

    // const double marker_size = 10.0;

    // auto points = plotter::get_random_points({-10,20}, {-10,20}, 100);
    //auto points2 = plotter::get_random_points({5,12}, {5,12}, 50);

    // plotter::keywords_t keywords {{"s", "20"}};
    // plotter::plot_points(points, marker_size);

    //plotter::plot_points({points, points2}, marker_size);

    /*auto func = std::function<double(double)>([](double x) -> double { return -x+3; });

    auto myplt = plotter::plotter::create(plotter::plottype::joint);

    myplt.plot_function(func, {-10,20}, {-10,20}, {"x", "y"});

    myplt.plot_points(points, func, marker_size);

    plotter::plotter::show();*/