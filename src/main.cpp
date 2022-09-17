#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glut.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int screen_height = 800;
int screen_width = 1280;

int button, state;

std::vector<std::pair<int, int>> points;
std::vector<std::vector<std::pair<int, int>>> splines;

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
    // if ((state == GLUT_DOWN) && (button == GLUT_LEFT_BUTTON)) {}
    if ((state == GLUT_UP) && (button == GLUT_LEFT_BUTTON)) {
        points.push_back(std::pair(x, y));
        std::cout << "Insert Point\t\t"
                  << "(" << x << ", " << y << ")" << std::endl;
        if (points.size() % 4 == 0) {
            splines.push_back(points);
        }
        std::cout << "Number of points:\t" << points.size() << std::endl;
        std::cout << "Number of splines:\t" << splines.size() << "\n"
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
