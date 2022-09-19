#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glut.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int screen_height = 800;
int screen_width = 1280;
int button, state;

int spline_order = 3;
int spline_subdiv = 200;
std::string spline_type = "Hermite";

typedef std::vector<std::pair<int, int>> PairVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> SplineMatrix;

PairVector points;
std::vector<SplineMatrix> splines;

unsigned int num_points = 0;
unsigned int num_splines = 0;
unsigned int num_control_points = spline_order + 1;

bool showConvexHull = false;
bool C1continuity = false;
bool C2continuity = false;

void CreateScreen() {
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition((1920 - screen_width) / 2,
                           (1080 - screen_height) / 2);
    glutInitWindowSize(screen_width, screen_height);
    glutCreateWindow("Spline Plotter");
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, screen_width, 0.0, screen_height);
}

void DrawPoint(int x, int y) {
    glPointSize(7);
    glColor3f(0.0f, 0.0f, 0.0f);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);

    glBegin(GL_POINTS);
    glVertex2i(x, y);
    glEnd();
}

void DrawText(int x, int y, std::string str) {
    glRasterPos2i(x + 5, y + 5);
    auto cstr = str.c_str();
    glutBitmapString(GLUT_BITMAP_HELVETICA_12,
                     reinterpret_cast<const unsigned char *>(cstr));
}

void DrawSpline(SplineMatrix spline) {
    glColor3f(1.0f, 0.0f, 0.0f);
    glLineWidth(2.0f);
    glEnable(GL_LINE_SMOOTH);

    glBegin(GL_LINE_STRIP);
    for (unsigned int i = 1; i < spline.rows(); i++) {
        glVertex2i((int)spline(i - 1, 0), (int)spline(i - 1, 1));
        glVertex2i((int)spline(i, 0), (int)spline(i, 1));
    }
    glEnd();
}

void DrawConvexHull(std::string spline_type) {
    glColor4f(0.2f, 0.2f, 0.5f, 0.2f);
    if (spline_type == "Hermite" || spline_type == "Bezier") {
        if (num_points >= num_control_points) {
            for (unsigned int i = 0; i < num_splines; i++) {
                glBegin(GL_POLYGON);
                for (int j = 0; j < 4; j++) {
                    glVertex2i(points[4 + 3 * i - j - 1].first,
                               points[4 + 3 * i - j - 1].second);
                }
                glEnd();
            }
        }
    } else if (spline_type == "BSpline" || spline_type == "CatmullRom") {
        if (num_points >= num_control_points) {
            for (unsigned int i = 0; i < num_splines; i++) {
                glBegin(GL_POLYGON);
                for (int j = 3; j >= 0; j--) {
                    glVertex2i(points[4 + i - j - 1].first,
                               points[4 + i - j - 1].second);
                }
                glEnd();
            }
        }
    }
}

SplineMatrix ComputeHermite(PairVector control_points) {
    assert(control_points.size() == num_control_points);

    Eigen::Matrix<double, 1, 4> t_vec;            // T
    Eigen::Matrix<double, 4, 4> basis_matrix;     // M
    Eigen::Matrix<double, 4, 2> geometry_matrix;  // G
    Eigen::Matrix<double, 1, 2> spline_point;
    SplineMatrix spline;  // Q

    spline_point << 0, 0;

    // clang-format off
    basis_matrix << 2, -2, 1, 1,
                    -3, 3, -2, -1,
                    0, 0, 1, 0,
                    1, 0, 0, 0;

    std::pair<int, int> tangent1 =
        std::make_pair(control_points[1].first - control_points[0].first,
                       control_points[1].second - control_points[0].second);
    std::pair<int, int> tangent2 =
        std::make_pair(control_points[2].first - control_points[3].first,
                       control_points[2].second - control_points[3].second);

    geometry_matrix << control_points[0].first, control_points[0].second,
        control_points[3].first, control_points[3].second, tangent1.first,
        tangent1.second, tangent2.first, tangent2.second;
    // clang-format on

    // Q = TMG
    for (double t = 0.0; t <= 1.0; t += 1.0 / spline_subdiv) {
        t_vec << pow(t, 3), pow(t, 2), t, 1;
        spline_point = t_vec * basis_matrix * geometry_matrix;
        spline.conservativeResize(spline.rows() + 1, spline.cols());
        spline.row(spline.rows() - 1) = spline_point;
    }

    assert(spline.rows() == spline_subdiv && spline.cols() == 2);
    return spline;
}

SplineMatrix ComputeBezier(PairVector control_points) {
    assert(control_points.size() == num_control_points);

    Eigen::Matrix<double, 1, 4> t_vec;            // T
    Eigen::Matrix<double, 4, 4> basis_matrix;     // M
    Eigen::Matrix<double, 4, 2> geometry_matrix;  // G
    Eigen::Matrix<double, 1, 2> spline_point;
    SplineMatrix spline;  // Q

    spline_point << 0, 0;

    // clang-format off
    basis_matrix << -1, 3, -3, 1,
                    3, -6, 3, 0,
                    -3, 3, 0, 0,
                    1, 0, 0, 0;

    geometry_matrix <<
        control_points[0].first, control_points[0].second,
        control_points[1].first, control_points[1].second,
        control_points[2].first, control_points[2].second,
        control_points[3].first, control_points[3].second;
    // clang-format on

    // Q = TMG
    for (double t = 0.0; t <= 1.0; t += 1.0 / spline_subdiv) {
        t_vec << pow(t, 3), pow(t, 2), t, 1;
        spline_point = t_vec * basis_matrix * geometry_matrix;
        spline.conservativeResize(spline.rows() + 1, spline.cols());
        spline.row(spline.rows() - 1) = spline_point;
    }

    assert(spline.rows() == spline_subdiv && spline.cols() == 2);
    return spline;
}

SplineMatrix ComputeBSpline(PairVector control_points) {
    assert(control_points.size() == num_control_points);

    Eigen::Matrix<double, 1, 4> t_vec;            // T
    Eigen::Matrix<double, 4, 4> basis_matrix;     // M
    Eigen::Matrix<double, 4, 2> geometry_matrix;  // G
    Eigen::Matrix<double, 1, 2> spline_point;
    SplineMatrix spline;  // Q

    spline_point << 0, 0;

    // clang-format off
    basis_matrix << -1, 3, -3, 1,
                    3, -6, 3, 0,
                    -3, 0, 3, 0,
                    1, 4, 1, 0;
    basis_matrix /= 6;

    geometry_matrix <<
        control_points[0].first, control_points[0].second,
        control_points[1].first, control_points[1].second,
        control_points[2].first, control_points[2].second,
        control_points[3].first, control_points[3].second;
    // clang-format on

    // Q = TMG
    for (double t = 0.0; t <= 1.0; t += 1.0 / spline_subdiv) {
        t_vec << pow(t, 3), pow(t, 2), t, 1;
        spline_point = t_vec * basis_matrix * geometry_matrix;
        spline.conservativeResize(spline.rows() + 1, spline.cols());
        spline.row(spline.rows() - 1) = spline_point;
    }

    assert(spline.rows() == spline_subdiv && spline.cols() == 2);
    return spline;
}

SplineMatrix ComputeCatmullRom(PairVector control_points) {
    assert(control_points.size() == num_control_points);

    Eigen::Matrix<double, 1, 4> t_vec;            // T
    Eigen::Matrix<double, 4, 4> basis_matrix;     // M
    Eigen::Matrix<double, 4, 2> geometry_matrix;  // G
    Eigen::Matrix<double, 1, 2> spline_point;
    SplineMatrix spline;  // Q

    spline_point << 0, 0;

    // clang-format off
    basis_matrix << -1, 3, -3, 1,
                    2, -5, 4, -1,
                    -1, 0, 1, 0,
                    0, 2, 0, 0;
    basis_matrix /= 2;

    geometry_matrix <<
        control_points[0].first, control_points[0].second,
        control_points[1].first, control_points[1].second,
        control_points[2].first, control_points[2].second,
        control_points[3].first, control_points[3].second;
    // clang-format on

    // Q = TMG
    for (double t = 0.0; t <= 1.0; t += 1.0 / spline_subdiv) {
        t_vec << pow(t, 3), pow(t, 2), t, 1;
        spline_point = t_vec * basis_matrix * geometry_matrix;
        spline.conservativeResize(spline.rows() + 1, spline.cols());
        spline.row(spline.rows() - 1) = spline_point;
    }

    assert(spline.rows() == spline_subdiv && spline.cols() == 2);
    return spline;
}

void EnforceContinuity() {
    // TODO
}

PairVector ReturnLastN(PairVector coordinates, unsigned int n) {
    PairVector::const_iterator first =
        coordinates.begin() + coordinates.size() - n;
    PairVector::const_iterator last = coordinates.begin() + coordinates.size();
    PairVector control_points(first, last);

    return control_points;
}

void GroupPoints(std::string spline_type) {
    num_points = points.size();

    if (spline_type == "Hermite" || spline_type == "Bezier") {
        if (num_points == num_control_points ||
            (num_points - 3 * num_splines == 4)) {
            PairVector control_points = ReturnLastN(points, num_control_points);
            if (spline_type == "Hermite") {
                splines.push_back(ComputeHermite(control_points));
            } else {
                splines.push_back(ComputeBezier(control_points));
            }
            num_splines++;
        }
    } else if (spline_type == "BSpline" || spline_type == "CatmullRom") {
        if (num_points >= num_control_points) {
            PairVector control_points = ReturnLastN(points, num_control_points);
            if (spline_type == "BSpline") {
                splines.push_back(ComputeBSpline(control_points));
            } else {
                splines.push_back(ComputeCatmullRom(control_points));
            }
            num_splines++;
        }
    }

    // std::cout << "Number of points:\t" << num_points << std::endl;
    // std::cout << "Number of splines:\t" << num_splines << std::endl
    //   << std::endl;
}

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    if (showConvexHull) DrawConvexHull(spline_type);

    int i = 1;
    for (std::pair<int, int> point : points) {
        DrawPoint(point.first, point.second);
        std::string point_string = 'P' + std::to_string(i);
        DrawText(point.first, point.second, point_string);
        i++;
    }

    for (SplineMatrix spline : splines) {
        DrawSpline(spline);
    }

    glFlush();
    glutSwapBuffers();
}

void ProcessMouse(int button, int state, int x, int y) {
    // click motion
    y = screen_height - y;
    if ((state == GLUT_UP) && (button == GLUT_LEFT_BUTTON)) {
        points.push_back(std::pair(x, y));
        std::cout << "Insert Point\t\t"
                  << "(" << x << ", " << y << ")" << std::endl;
        GroupPoints(spline_type);
    }
}

void ProcessMouseActiveMotion(int x, int y) {
    // drag motion
    if ((state == GLUT_DOWN) && (button == GLUT_LEFT_BUTTON)) {
    }
}

bool CheckArgSplineType(std::vector<std::string> args) {
    bool valid = false;
    auto checkSplineType =
        std::find(std::begin(args), std::end(args), "--spline_type");
    if (checkSplineType != std::end(args)) {
        spline_type = *(++checkSplineType);
        if (spline_type == "Hermite" || spline_type == "Bezier" ||
            spline_type == "BSpline" || spline_type == "CatmullRom") {
            valid = true;
        } else {
            std::cout
                << "Invalid argument --spline_type {Hermite, Bezier, BSpline, "
                   "CatmullRom}"
                << std::endl;
        }
    } else {
        std::cout << "No argument --spline_type {Hermite, Bezier, BSpline, "
                     "CatmullRom}"
                  << std::endl;
    }
    return valid;
}

void CheckArgConvexHull(std::vector<std::string> args) {
    auto checkConvexHullPlotting =
        std::find(std::begin(args), std::end(args), "--show_convex_hull");
    if (checkConvexHullPlotting != std::end(args)) {
        if (*(++checkConvexHullPlotting) == "true") showConvexHull = true;
    }
}

int main(int argc, char *argv[]) {
    std::vector<std::string> args(argv, argv + argc);
    if(CheckArgSplineType(args) == false) return EXIT_FAILURE;
    CheckArgConvexHull(args);

    glutInit(&argc, argv);
    CreateScreen();
    glutDisplayFunc(RenderScene);
    glutIdleFunc(RenderScene);

    glutMouseFunc(ProcessMouse);
    glutMotionFunc(ProcessMouseActiveMotion);

    glutMainLoop();
    return EXIT_SUCCESS;
}
