#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <fmt/format.h>

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

const int screen_height = 800;
const int screen_width = 1280;

unsigned int spline_degree = 3;  // p
unsigned int spline_subdiv = 150;
std::string spline_type = "Hermite";

typedef std::vector<std::pair<int, int>> PairVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> SplineMatrix;

PairVector points;
std::vector<SplineMatrix> splines;

unsigned int num_points = 0;
unsigned int num_splines = 0;
unsigned int num_control_points = spline_degree + 1;

unsigned int showConvexHull = 0;
unsigned int GCont = 0;
unsigned int CCont = 0;

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

void DrawText(int x, int y, std::string str, void* font = GLUT_BITMAP_9_BY_15) {
    glColor3f(0.0f, 0.0f, 0.0f);
    glRasterPos2i(x + 5, y + 5);
    auto cstr = str.c_str();
    glutBitmapString(font, reinterpret_cast<const unsigned char*>(cstr));
}

void DrawSpline(SplineMatrix& spline) {
    glColor3f(1.0f, 0.0f, 0.0f);
    glLineWidth(2.5f);
    glEnable(GL_LINE_SMOOTH);

    glBegin(GL_LINE_STRIP);
    for (unsigned int i = 1; i < spline.rows(); i++) {
        glVertex2i(static_cast<int>(spline(i - 1, 0)),
                   static_cast<int>(spline(i - 1, 1)));
        glVertex2i(static_cast<int>(spline(i, 0)),
                   static_cast<int>(spline(i, 1)));
    }
    glEnd();
}

void DrawSimplex(std::string spline_type_, unsigned int style = 0) {
    // Draw lines or polygons to illustrate the simplex ("control polygon")
    if (num_points >= num_control_points) {
        switch (style) {
            case (1):
                glPushAttrib(GL_ENABLE_BIT);
                glLineStipple(4, 0xAAAA);
                glEnable(GL_LINE_STIPPLE);
                glColor3f(0.0f, 0.0f, 0.0f);
                glLineWidth(1.5f);
                glBegin(GL_LINES);

                for (unsigned int i = 0; i < num_points - 1; i++) {
                    glVertex2i(points[i].first, points[i].second);
                    glVertex2i(points[i + 1].first, points[i + 1].second);
                }
                glEnd();
                glPopAttrib();
                break;
            case (2):
                if (spline_type_ == "Hermite" || spline_type_ == "Bezier" ||
                    spline_type == "MINVO") {
                    for (unsigned int i = 0; i < num_splines; i++) {
                        glPushAttrib(GL_ENABLE_BIT);
                        glColor4f(0.2f, 0.5f, 0.2f, 0.2f);
                        glBegin(GL_POLYGON);

                        for (unsigned int j = 0; j < num_control_points; j++) {
                            glVertex2i(points[num_control_points +
                                              spline_degree * i - j - 1]
                                           .first,
                                       points[num_control_points +
                                              spline_degree * i - j - 1]
                                           .second);
                        }
                        glPopAttrib();
                        glEnd();
                    }
                } else if (spline_type_ == "BSpline" ||
                           spline_type_ == "CatmullRom") {
                    for (unsigned int i = 0; i < num_splines; i++) {
                        glPushAttrib(GL_ENABLE_BIT);
                        glColor4f(0.2f, 0.5f, 0.2f, 0.2f);
                        glBegin(GL_POLYGON);
                        for (unsigned int j = 0; j < num_control_points; j++) {
                            glVertex2i(
                                points[num_control_points + i - j - 1].first,
                                points[num_control_points + i - j - 1].second);
                        }
                        glPopAttrib();
                        glEnd();
                    }
                }
                break;
            default:
                std::cout << "Invalid convex hull plotting option" << std::endl;
                break;
        }
    }
}

void DisplayData(std::string spline_type_, unsigned int num_points_,
                 unsigned int CCont_, unsigned int GCont_) {
    auto font = GLUT_BITMAP_HELVETICA_12;

    std::string display_spline_type = "Spline type: " + spline_type_;
    DrawText(10, screen_height - 30 - 25 * 0, display_spline_type, font);

    std::string display_points =
        fmt::format("Number of points: {}", num_points_);
    DrawText(10, screen_height - 30 - 25 * 1, display_points, font);

    std::string display_cont =
        fmt::format("Continuity: C{}, G{}", CCont_, GCont_);
    DrawText(10, screen_height - 30 - 25 * 2, display_cont, font);
}

SplineMatrix ComputeSpline(PairVector& control_points, std::string spline_type_,
                           unsigned int spline_degree_,
                           unsigned int derivative) {
    int num_control_points_ = static_cast<int>(spline_degree_) + 1;
    assert(static_cast<int>(control_points.size()) == num_control_points_);

    // declare dynamic size matrices as spline degree not known at compile time
    // allocate array of coefficients with constructors that take size
    // allocates on the heap
    SplineMatrix spline;                                             // Q
    Eigen::Matrix<double, 1, -1> param_vec(1, num_control_points_);  // T^(p)
    Eigen::Matrix<double, -1, -1> basis_matrix(num_control_points_,
                                               num_control_points_);        // M
    Eigen::Matrix<double, -1, -1> geometry_matrix(num_control_points_, 2);  // G

    Eigen::Matrix<double, 1, 2> spline_point(0.0, 0.0);

    if (spline_type_ == "Hermite") {
        basis_matrix << 2, -2, 1, 1, -3, 3, -2, -1, 0, 0, 1, 0, 1, 0, 0, 0;

        std::pair<int, int> tangent1 =
            std::make_pair(control_points[1].first - control_points[0].first,
                           control_points[1].second - control_points[0].second);
        std::pair<int, int> tangent2 =
            std::make_pair(control_points[3].first - control_points[2].first,
                           control_points[3].second - control_points[2].second);

        geometry_matrix << control_points[0].first, control_points[0].second,
            control_points[3].first, control_points[3].second, tangent1.first,
            tangent1.second, tangent2.first, tangent2.second;
    } else {
        geometry_matrix << control_points[0].first, control_points[0].second,
            control_points[1].first, control_points[1].second,
            control_points[2].first, control_points[2].second,
            control_points[3].first, control_points[3].second;

        if (spline_type_ == "Bezier") {
            basis_matrix << -1, 3, -3, 1, 3, -6, 3, 0, -3, 3, 0, 0, 1, 0, 0, 0;
        } else if (spline_type_ == "BSpline") {
            basis_matrix << -1, 3, -3, 1, 3, -6, 3, 0, -3, 0, 3, 0, 1, 4, 1, 0;
            basis_matrix /= 6;
        } else if (spline_type_ == "CatmullRom") {
            basis_matrix << -1, 3, -3, 1, 2, -5, 4, -1, -1, 0, 1, 0, 0, 2, 0, 0;
            basis_matrix /= 2;
        } else if (spline_type_ == "MINVO") {
            basis_matrix << -0.4302, 0.4568, -0.02698, 0.0004103, 0.8349,
                -0.4568, -0.7921, 0.4996, -0.8349, -0.4568, 0.7921, 0.4996,
                0.4302, 0.4568, 0.02698, 0.0004103;
            basis_matrix.transposeInPlace();
        }
    }

    double t_range[2] = {0.0, 1.0};
    if (spline_type_ == "MINVO") t_range[0] = -1.0;

    // Q = TMG
    for (double t = t_range[0]; t <= t_range[1]; t += 1.0 / spline_subdiv) {
        for (unsigned int i = spline_degree_; spline_degree_ >= i; i--) {
            switch (derivative) {
                case 0:  // compute points of the spline
                    param_vec(i) = pow(t, i);
                    break;
                case 1:  // compute parametric velocity
                    param_vec(i) = i * pow(t, i - 1);
                    break;
                case 2:  // compute parametric acceleration
                    i*(i - 1) * pow(t, i - 2);
                    break;
                default:
                    std::cout << "Choose from derivatives = {0, 1, 2}"
                              << std::endl;
                    break;
            }
        }
        param_vec.reverseInPlace();  // t parameter ordered in descending powers
        spline_point = param_vec * basis_matrix * geometry_matrix;
        spline.conservativeResize(spline.rows() + 1, spline.cols());
        spline.row(spline.rows() - 1) = spline_point;
    }
    return spline;
}

PairVector ReturnLastN(PairVector& coordinates, unsigned int n) {
    unsigned int lenInput = static_cast<unsigned int>(coordinates.size());
    PairVector::const_iterator init = coordinates.begin() + lenInput - n;
    PairVector::const_iterator last = coordinates.begin() + lenInput;
    PairVector control_points(init, last);

    return control_points;
}

unsigned int ReturnPointIndex(std::string spline_type_) {
    unsigned int index = num_points;

    if (num_points >= num_control_points) {
        if (spline_type_ == "Hermite" || spline_type_ == "Bezier" ||
            spline_type_ == "MINVO") {
            index = (num_points - 1) % spline_degree;
        } else {
            index = 0;
        }
    }
    return index;
}

void EnforceContinuity(PairVector& coordinates, unsigned int GCont_,
                       unsigned int CCont_) {
    Eigen::Vector2d prevDir;
    Eigen::Vector2d currDir;
    Eigen::Vector2d prevPoint;
    Eigen::Vector2d newPoint;

    // Enforce G1 continuity
    if (GCont_ == 1 && (spline_type == "Hermite" || spline_type == "Bezier")) {
        if (ReturnPointIndex(spline_type) == 1 &&
            num_points > num_control_points) {
            // indices further reduced by 1 as num_points starts from 1
            prevDir << static_cast<double>(
                coordinates[num_points - 1 - 1].first -
                coordinates[num_points - 2 - 1].first),
                static_cast<double>(coordinates[num_points - 1 - 1].second -
                                    coordinates[num_points - 2 - 1].second);

            currDir << static_cast<double>(
                coordinates[num_points - 1].first -
                coordinates[num_points - 1 - 1].first),
                static_cast<double>(coordinates[num_points - 1].second -
                                    coordinates[num_points - 1 - 1].second);

            prevPoint << static_cast<double>(
                coordinates[num_points - 1 - 1].first),
                static_cast<double>(coordinates[num_points - 1 - 1].second);

            double vel = currDir.dot(prevDir.normalized());
            // Enforce C1 continuity
            if (CCont_ == 1) vel = prevDir.norm();

            prevDir = prevDir.normalized();
            newPoint = vel * prevDir + prevPoint;
            newPoint(0) = static_cast<int>(newPoint(0));
            newPoint(1) = static_cast<int>(newPoint(1));

            std::cout << "Adjust Point " << num_points << "\t\t"
                      << "(" << newPoint(0) << ", " << newPoint(1) << ")"
                      << std::endl;

            coordinates[num_points - 1].first = (newPoint(0));
            coordinates[num_points - 1].second = (newPoint(1));
        }
    }
}

void GroupPoints(std::string spline_type_) {
    if (spline_type_ == "Hermite" || spline_type_ == "Bezier" ||
        spline_type == "MINVO") {
        if (ReturnPointIndex(spline_type_) == 0) {
            PairVector control_points = ReturnLastN(points, num_control_points);
            splines.push_back(
                ComputeSpline(control_points, spline_type_, spline_degree, 0));
            num_splines++;
        }
    } else if (spline_type_ == "BSpline" || spline_type_ == "CatmullRom") {
        if (ReturnPointIndex(spline_type_) == 0) {
            PairVector control_points = ReturnLastN(points, num_control_points);
            splines.push_back(
                ComputeSpline(control_points, spline_type_, spline_degree, 0));
            num_splines++;
        }
    }
}

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Convex Hull drawn in first "layer" below points and splines
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    if (showConvexHull) DrawSimplex(spline_type, showConvexHull);

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

    DisplayData(spline_type, num_points, CCont, GCont);

    glFlush();
    glutSwapBuffers();
}

void RemoveAllPoints() {
    std::cout << "Remove all inserted points" << std::endl;
    points.clear();
    splines.clear();
    num_points = 0;
    num_splines = 0;
}

void RemovePrevPoint() {
    std::cout << "Remove Point " << num_points << std::endl;
    points.pop_back();
    num_points--;
    if (spline_type == "Hermite" || spline_type == "Bezier" ||
        spline_type == "MINVO") {
        if (num_points == num_control_points - 1 ||
            ((num_points - num_control_points) % spline_degree ==
                 (spline_degree - 1) &&
             num_points >= num_control_points)) {
            num_splines--;
            splines.pop_back();
        }
    } else {
        if (num_points >= num_control_points - 1) {
            splines.pop_back();
            num_splines--;
        }
    }
}

void ProcessNormalKeyPress(unsigned char key, int x, int y) {
    // keyboard input (normal keys)
    switch (key) {
        case 'r':
            RemovePrevPoint();
            break;
        default:
            break;
    }
}

void ProcessSpecialKeyPress(int key, int x, int y) {
    // keyboard input (special keys)
    switch (key) {
        case GLUT_KEY_F1:
            RemoveAllPoints();
            break;
        default:
            break;
    }
}

void ProcessKeyRelease(int key, int x, int y) {
    // keyboard release
}

void ProcessMouse(int button, int state, int x, int y) {
    // click motion
    y = screen_height - y;
    if ((state == GLUT_UP) && (button == GLUT_LEFT_BUTTON)) {
        points.push_back(std::pair(x, y));
        num_points = points.size();
        std::cout << "Insert Point " << num_points << "\t\t"
                  << "(" << x << ", " << y << ")" << std::endl;
        EnforceContinuity(points, GCont, CCont);
        GroupPoints(spline_type);
    }
}

void ProcessMouseActiveMotion(int x, int y) {
    // drag motion
    // if ((state == GLUT_DOWN) && (button == GLUT_LEFT_BUTTON)) {
    // }
}

bool CheckArgSplineType(std::vector<std::string> args) {
    bool valid = false;
    auto checkSplineType =
        std::find(std::begin(args), std::end(args), "--spline_type");
    if (checkSplineType != std::end(args)) {
        std::string spline_type_ = *(++checkSplineType);
        if (spline_type_ == "Hermite" || spline_type_ == "Bezier" ||
            spline_type_ == "BSpline" || spline_type_ == "CatmullRom" ||
            spline_type_ == "MINVO") {
            valid = true;
            spline_type = spline_type_;
        } else {
            std::cout
                << "Invalid argument --spline_type {Hermite, Bezier, BSpline, "
                   "CatmullRom, MINVO}"
                << std::endl;
        }
    } else {
        std::cout << "No argument --spline_type {Hermite, Bezier, BSpline, "
                     "CatmullRom, MINVO}"
                  << std::endl;
    }
    return valid;
}

void CheckArgConvexHull(std::vector<std::string> args) {
    auto checkConvexHullPlotting =
        std::find(std::begin(args), std::end(args), "--show_convex_hull");
    if (checkConvexHullPlotting != std::end(args)) {
        showConvexHull = std::stoul(*(++checkConvexHullPlotting));
    }
}

void CheckArgContinuity(std::vector<std::string> args) {
    bool C2_spline = (spline_type == "BSpline" || spline_type == "CatmullRom");

    if (C2_spline) {
        std::cout << "C2 spline chosen: setting continuity C2, G2" << std::endl;
        CCont = 2, GCont = 2;
    } else {
        auto checkGCont =
            std::find(std::begin(args), std::end(args), "--GCont");
        auto checkCCont =
            std::find(std::begin(args), std::end(args), "--CCont");

        if (checkCCont != std::end(args)) {
            CCont = std::stoul(*(++checkCCont));
            if (checkGCont != std::end(args)) {
                GCont = std::stoul(*(++checkGCont));
            }
            if (GCont > CCont) {
                std::cout << "Invalid continuity specified (GCont > CCont): "
                             "setting GCont = CCont"
                          << std::endl;
                GCont = CCont;
            }
            std::cout << fmt::format("Setting continuity C{}, G{}", CCont,
                                     GCont)
                      << std::endl;
        } else if (checkGCont != std::end(args)) {
            GCont = std::stoul(*(++checkGCont));
            std::cout << fmt::format("Setting continuity C{}, G{}", CCont,
                                     GCont)
                      << std::endl;
        } else {
            std::cout << "--CCont not specified. Defaulting to C0, G0"
                      << std::endl;
            CCont = 0;
            GCont = 0;
        }
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv, argv + argc);
    // required arguments
    if (CheckArgSplineType(args) == false) return EXIT_FAILURE;

    // optional arguments
    CheckArgConvexHull(args);
    CheckArgContinuity(args);

    glutInit(&argc, argv);
    CreateScreen();
    glutDisplayFunc(RenderScene);
    glutIdleFunc(RenderScene);

    glutMouseFunc(ProcessMouse);
    glutMotionFunc(ProcessMouseActiveMotion);

    glutIgnoreKeyRepeat(1);
    glutKeyboardFunc(ProcessNormalKeyPress);
    glutSpecialFunc(ProcessSpecialKeyPress);

    glutMainLoop();
    return EXIT_SUCCESS;
}
