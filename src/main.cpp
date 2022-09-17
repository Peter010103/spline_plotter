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

std::vector<std::pair<int, int>> points;

void CreateScreen() {
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(350, 120);
    glutInitWindowSize(screen_width, screen_height);
    glutCreateWindow("Spline Plotter");
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, screen_width, 0.0, screen_height);
}

void DrawPoint(int x, int y) {
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

void ComputHermite(std::vector<std::pair<int, int>> control_points) {
    assert(control_points.size() == 4);

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

void ComputeBezier(std::vector<std::pair<int, int>> control_points) {
    assert(control_points.size() == 4);

    Eigen::Matrix<double, 1, 4> t_vec;            // T
    Eigen::Matrix<double, 4, 4> basis_matrix;     // M
    Eigen::Matrix<double, 4, 2> geometry_matrix;  // G
    Eigen::Matrix<double, 1, 2> spline_point;
    Eigen::Matrix<double, -1, 2> spline;  // Q

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
    for (double t = 0.0; t <= 1.0; t += 0.005) {
        t_vec << pow(t, 3), pow(t, 2), t, 1;  // T (row-vector)
        spline_point = t_vec * basis_matrix * geometry_matrix;
        // std::cout << spline_point << std::endl << std::endl;
        spline.conservativeResize(spline.rows() + 1, spline.cols());
        spline.row(spline.rows() - 1) = spline_point;
    }

    assert(spline.rows() == 200 && spline.cols() == 2);
}

void ComputeBSpline(std::vector<std::pair<int, int>> control_points) {
    assert(control_points.size() == 4);

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
    assert(control_points.size() == 4);

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

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPointSize(7);
    glColor3f(0.0f, 0.0f, 0.0f);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);

    int i = 1;
    for (std::pair<int, int> point : points) {
        DrawPoint(point.first, point.second);
        std::string point_string = 'P' + std::to_string(i);
        DrawText(point.first, point.second, point_string);
        i++;
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

        if (points.size() % 4 == 0) {
            std::vector<std::pair<int, int>>::const_iterator first =
                points.begin() + points.size() - 4;
            std::vector<std::pair<int, int>>::const_iterator last =
                points.begin() + points.size();
            std::vector<std::pair<int, int>> control_points(first, last);
            ComputeBezier(control_points);
        }
        std::cout << "Number of points:\t" << points.size() << std::endl;
        std::cout << "Number of splines:\t" << points.size() / 4 << std::endl
                  << std::endl;
    }
}

void ProcessMouseActiveMotion(int x, int y) {
    // drag motion
    if ((state == GLUT_DOWN) && (button == GLUT_LEFT_BUTTON)) {
        std::cout << "Currently dragging" << std::endl;
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
