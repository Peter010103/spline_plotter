#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glut.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int screen_height = 800;
int screen_width = 1280;

int button, state;

int spline_order = 3;
int spline_subdiv = 200;

typedef Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> SplineMatrix;
std::vector<std::pair<int, int>> points;
std::vector<SplineMatrix> splines;

unsigned int num_points = 0;
unsigned int num_splines = 0;
unsigned int num_control_points = spline_order + 1;

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

void DrawConvexHull() {
    // TODO
}

void ComputHermite(std::vector<std::pair<int, int>> control_points) {
    assert(control_points.size() == num_control_points;

    Eigen::Matrix<double, 1, 4> t_vec;            // T
    Eigen::Matrix<double, 4, 4> basis_matrix;     // M
    Eigen::Matrix<double, 4, 2> geometry_matrix;  // G
    Eigen::Matrix<double, 1, 2> spline_point;

    // clang-format off
    basis_matrix << 2, -2, 1, 1,
                    -3, 3, -2, -1,
                    0, 0, 1, 0,
                    1, 0, 0, 0;

    spline_point << 0, 0;
    // clang-format on
}

SplineMatrix ComputeBezier(std::vector<std::pair<int, int>> control_points) {
    assert(control_points.size() == num_control_points);

    Eigen::Matrix<double, 1, 4> t_vec;            // T
    Eigen::Matrix<double, 4, 4> basis_matrix;     // M
    Eigen::Matrix<double, 4, 2> geometry_matrix;  // G
    Eigen::Matrix<double, 1, 2> spline_point;
    SplineMatrix spline;  // Q

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

    spline_point << 0, 0;
    // clang-format on

    // Q = TMG
    for (double t = 0.0; t <= 1.0; t += 1.0 / spline_subdiv) {
        t_vec << pow(t, 3), pow(t, 2), t, 1;
        spline_point = t_vec * basis_matrix * geometry_matrix;
        // std::cout << spline_point << std::endl << std::endl;
        spline.conservativeResize(spline.rows() + 1, spline.cols());
        spline.row(spline.rows() - 1) = spline_point;
    }

    assert(spline.rows() == spline_subdiv && spline.cols() == 2);
    return spline;
}

void ComputeBSpline(std::vector<std::pair<int, int>> control_points) {
    assert(control_points.size() == num_control_points);

    Eigen::Matrix<double, 1, 4> t_vec;            // T
    Eigen::Matrix<double, 4, 4> basis_matrix;     // M
    Eigen::Matrix<double, 4, 2> geometry_matrix;  // G
    Eigen::Matrix<double, 1, 2> spline_point;

    // clang-format off
    basis_matrix << -1, 3, -3, 1,
                    3, -6, 3, 0,
                    -3, 0, 3, 0,
                    1, 4, 1, 0;
    basis_matrix /= 6;

    spline_point << 0, 0;
    // clang-format on
}

void ComputeCatmullRom(std::vector<std::pair<int, int>> control_points) {
    assert(control_points.size() == num_control_points);

    Eigen::Matrix<double, 1, 4> t_vec;            // T
    Eigen::Matrix<double, 4, 4> basis_matrix;     // M
    Eigen::Matrix<double, 4, 2> geometry_matrix;  // G
    Eigen::Matrix<double, 1, 2> spline_point;

    // clang-format off
    basis_matrix << -1, 3, -3, 1,
                    2, -5, 4, 1,
                    -1, 0, 1, 0,
                    0, 2, 0, 0;
    basis_matrix /= 2;

    spline_point << 0, 0;
    // clang-format on
}

void EnforceContinuity() {
    // TODO
}

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

        num_points = points.size();
        if (num_points == num_control_points ||
            ((num_points - num_control_points) % spline_order == 0 &&
             num_points > num_control_points)) {
            std::vector<std::pair<int, int>>::const_iterator first =
                points.begin() + points.size() - num_control_points;
            std::vector<std::pair<int, int>>::const_iterator last =
                points.begin() + points.size();
            std::vector<std::pair<int, int>> control_points(first, last);
            splines.push_back(ComputeBezier(control_points));
            num_splines++;
        }
        std::cout << "Number of points:\t" << num_points << std::endl;
        std::cout << "Number of splines:\t" << num_splines << std::endl
                  << std::endl;
    }
}

void ProcessMouseActiveMotion(int x, int y) {
    // drag motion
    if ((state == GLUT_DOWN) && (button == GLUT_LEFT_BUTTON)) {
    }
}

int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
    CreateScreen();
    glutDisplayFunc(RenderScene);
    glutIdleFunc(RenderScene);

    glutMouseFunc(ProcessMouse);
    glutMotionFunc(ProcessMouseActiveMotion);

    glutMainLoop();
    return EXIT_SUCCESS;
}
