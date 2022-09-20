#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glut.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

const int screen_height = 800;
const int screen_width = 1280;

unsigned int spline_order = 3;
unsigned int spline_subdiv = 150;
std::string spline_type = "Hermite";

typedef std::vector<std::pair<int, int>> PairVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> SplineMatrix;

PairVector points;
std::vector<SplineMatrix> splines;

unsigned int num_points = 0;
unsigned int num_splines = 0;
unsigned int num_control_points = spline_order + 1;

unsigned int showConvexHull = 0;
unsigned int GCont = 1;
unsigned int CCont = 1;

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
    glutBitmapString(GLUT_BITMAP_9_BY_15,
                     reinterpret_cast<const unsigned char*>(cstr));
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

void DrawConvexHull(std::string spline_type_, unsigned int style = 0) {
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

                        for (unsigned int j = 0; j < 4; j++) {
                            glVertex2i(points[4 + 3 * i - j - 1].first,
                                       points[4 + 3 * i - j - 1].second);
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
                        for (unsigned int j = 0; j < 4; j++) {
                            glVertex2i(points[4 + i - j - 1].first,
                                       points[4 + i - j - 1].second);
                        }
                        glPopAttrib();
                        glEnd();
                    }
                }
                break;
            default:
                break;
        }
    }
}

SplineMatrix ComputeSplinePoints(PairVector& control_points,
                                 std::string spline_type_,
                                 unsigned int spline_order_) {
    int num_control_points_ = static_cast<int>(spline_order_) + 1;
    assert(static_cast<int>(control_points.size()) == num_control_points_);

    Eigen::Matrix<double, 1, -1> t_vec(1, num_control_points_);  // T
    Eigen::Matrix<double, -1, -1> basis_matrix(num_control_points_,
                                               num_control_points_);        // M
    Eigen::Matrix<double, -1, -1> geometry_matrix(num_control_points_, 2);  // G
    Eigen::Matrix<double, 1, 2> spline_point;
    SplineMatrix spline;  // Q

    spline_point << 0, 0;

    if (spline_type_ == "Hermite") {
        basis_matrix << 2, -2, 1, 1, -3, 3, -2, -1, 0, 0, 1, 0, 1, 0, 0, 0;

        std::pair<int, int> tangent1 =
            std::make_pair(control_points[1].first - control_points[0].first,
                           control_points[1].second - control_points[0].second);
        std::pair<int, int> tangent2 =
            std::make_pair(control_points[2].first - control_points[3].first,
                           control_points[2].second - control_points[3].second);

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

    // Q = TMG
    if (spline_type_ == "MINVO")
        for (double t = -1.0; t <= 1.0; t += 1.0 / spline_subdiv) {
            t_vec << pow(t, 3), pow(t, 2), t, 1;
            spline_point = t_vec * basis_matrix * geometry_matrix;
            spline.conservativeResize(spline.rows() + 1, spline.cols());
            spline.row(spline.rows() - 1) = spline_point;
        }
    else {
        for (double t = 0.0; t <= 1.0; t += 1.0 / spline_subdiv) {
            t_vec << pow(t, 3), pow(t, 2), t, 1;
            spline_point = t_vec * basis_matrix * geometry_matrix;
            spline.conservativeResize(spline.rows() + 1, spline.cols());
            spline.row(spline.rows() - 1) = spline_point;
        }
    }
    return spline;
}

PairVector ReturnLastN(PairVector& coordinates, unsigned int n) {
    unsigned int lenInput = static_cast<unsigned int>(coordinates.size());
    PairVector::const_iterator first = coordinates.begin() + lenInput - n;
    PairVector::const_iterator last = coordinates.begin() + lenInput;
    PairVector control_points(first, last);

    return control_points;
}

void EnforceContinuity(PairVector& coordinates, unsigned int GCont_,
                       unsigned int CCont_) {
    Eigen::Vector2d prevDir;
    Eigen::Vector2d currDir;
    Eigen::Vector2d prevPoint;
    Eigen::Vector2d newPoint;

    // Enforce G1 continuity
    if (GCont_ == 1 && (spline_type == "Hermite" || spline_type == "Bezier" ||
                        spline_type == "MINVO")) {
        if (num_points % 3 == 2 && num_points > num_control_points) {
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

            prevDir = prevDir.normalized();
            newPoint = (currDir.dot(prevDir)) * prevDir + prevPoint;
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
        if (num_points == num_control_points ||
            (num_points - 3 * num_splines == 4)) {
            PairVector control_points = ReturnLastN(points, num_control_points);
            splines.push_back(ComputeSplinePoints(control_points, spline_type_,
                                                  spline_order));
            num_splines++;
        }
    } else if (spline_type_ == "BSpline" || spline_type_ == "CatmullRom") {
        if (num_points >= num_control_points) {
            PairVector control_points = ReturnLastN(points, num_control_points);
            splines.push_back(ComputeSplinePoints(control_points, spline_type_,
                                                  spline_order));
            num_splines++;
        }
    }
}

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Convex Hull drawn in first "layer" below points and splines
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    if (showConvexHull) DrawConvexHull(spline_type, showConvexHull);

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

int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv, argv + argc);
    if (CheckArgSplineType(args) == false) return EXIT_FAILURE;
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
